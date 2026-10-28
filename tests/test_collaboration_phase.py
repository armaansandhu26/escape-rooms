import json
from pathlib import Path
from threading import Barrier

import pytest
import yaml

from motiverooms.actions import parse_action
from motiverooms.azure_openai import AzureOpenAISettings
from motiverooms.env import MotiveRoomsEnv
from motiverooms.evaluation import load_suite
from motiverooms.experiment import independent_actions, read_events, replay
from motiverooms.generator import generate_room
from motiverooms.llm_policy import LLMAgentPolicy
from scripts.evaluate import execute_plan, make_plan
from scripts.summarize_runs import summarize_batch


@pytest.mark.parametrize("line", ["say(secret 1234)", "write_note(secret 1234)", "read_notes()", "give(key)"])
@pytest.mark.parametrize("joint", [True, False])
def test_disabled_channels_are_enforced_and_cost_turns(line, joint):
    env = MotiveRoomsEnv(generate_room("private_handoffs", 0).room, communication="disabled")
    env.engine.agents["A"].bag.add("key")
    reason = env.engine.action_rejection_reason("A", parse_action(line))
    assert "disabled" in reason
    before = env.engine.agents["A"].turns_left
    outcome = (env.step_joint({"A": line, "B": "wait()"})["A"] if joint
               else env.engine.step("A", line))
    assert not outcome.valid and "disabled" in outcome.result
    assert env.engine.agents["A"].turns_left == before - 1
    assert env.engine.agents["A"].bag == {"key"}
    assert not env.engine.agents["B"].bag
    assert not env.engine.messages and not env.engine.notes


def test_private_clues_and_current_choices_are_not_exposed():
    case = generate_room("private_handoffs", 0)
    env = MotiveRoomsEnv(case.room)
    handoffs = {h["holder"]: h for h in case.metadata["handoffs"]}
    clue = next(n for n in case.room.nodes_for_agent("A") if n.kind == "clue" and n.code == handoffs["A"]["resource"])
    secret = case.room.codes[clue.code]
    env.step_joint({"A": f"inspect({clue.id})", "B": "wait()"})
    assert secret in env.observe("A")
    assert secret not in env.observe("B")
    assert f"inspect({clue.id})" not in env.observe("B")
    gate = Barrier(2, timeout=5)
    observations = env.observe_all()

    class Policy:
        def __init__(self, agent):
            self.agent = agent

        def choose_action(self, observation):
            assert observation == observations[self.agent]
            gate.wait() # Both calls must start before either can return a choice.
            return "wait()", "wait()"

    assert independent_actions({a: Policy(a) for a in env.agent_ids}, observations, env) == {"A": "wait()", "B": "wait()"}


def config():
    return yaml.safe_load(Path("config/eval_collaboration_pilot.yaml").read_text())


def test_phase2_plan_has_six_unique_pairs_and_alternates_order():
    plan = make_plan(config(), load_suite("rooms/diagnostics/manifest.json"))
    assert len(plan) == 12
    assert {p[2] for p in plan} == {"self_play"}
    assert {p[3] for p in plan} == {"A"} # No duplicate self-play seat evaluations.
    assert [p[1]["communication"] for p in plan[:4]] == ["enabled", "disabled", "disabled", "enabled"]
    assert sum(2 * p[1]["recommended_turn_budget"] for p in plan) == 224
    for first, second in zip(plan[::2], plan[1::2]):
        assert first[0] == second[0]
        assert first[4] == second[4]


def test_offline_controls_replay_and_rebuild_without_azure(tmp_path):
    cfg = config()
    plan = make_plan(cfg, load_suite("rooms/diagnostics/manifest.json"), offline=True)
    folder = tmp_path / "controls"
    records = execute_plan(cfg, plan, folder, offline=True)
    assert len(records) == 12
    assert all(r["metrics"]["joint_escape"] and r["metrics"]["api_calls"] == 0 for r in records)
    for record in records:
        assert replay(record["path"]) == record["metrics"]
        if record["metadata"]["communication"] == "disabled":
            assert all(s["messages_sent"] == s["notes_written"] == s["used_shared_resources"] == 0
                       for s in record["metrics"]["per_agent"].values())
    pairs = json.loads((folder / "communication_comparison.json").read_text())
    assert pairs["completed_matched_pairs"] == pairs["planned_pairs"] == 6
    assert {p["steps_saved_with_communication"] for p in pairs["pairs"]} == {0, 3, 5}
    report = (folder / "summary.md").read_text()
    assert "Offline systems check" in report and "6/6 planned" in report
    assert summarize_batch(folder).read_text() == report


def test_mocked_two_ai_agents_use_separate_clients_and_reports_count_both(tmp_path, monkeypatch):
    cfg = config()
    cfg["communication_instruction"] = "explained"
    cfg["action_interface"] = "text"
    plan = make_plan(cfg, load_suite("rooms/diagnostics/manifest.json"))[:2]
    clients = []

    class Settings:
        @staticmethod
        def from_env():
            return AzureOpenAISettings("placeholder", "https://example.com/openai/v1/", "test")

    class Client:
        def __init__(self, settings, budget):
            self.settings, self.budget = settings, budget
            self.prompts = []
            clients.append(self)

        def complete(self, *, instructions, user_input):
            self.prompts.append((instructions, user_input))
            return "say(hello)" # Accepted on, rejected off; no retries or correction.

    monkeypatch.setattr("scripts.evaluate.AzureOpenAISettings", Settings)
    folder = tmp_path / "mock"
    records = execute_plan(cfg, plan, folder, client_factory=Client)
    assert len(clients) == 4 and len({id(c.budget) for c in clients}) == 1
    assert all(r["metadata"]["policies"] == {"A": "azure", "B": "azure"} for r in records)
    assert clients[0].prompts[0][0].startswith("You are agent A.")
    assert "COMMUNICATION FOR THIS ATTEMPT" in clients[0].prompts[0][0]
    assert all(r["metadata"]["communication_instruction"] == "explained" for r in records)
    assert clients[1].prompts[0][0].startswith("You are agent B.")
    assert "say(<message>)" in clients[0].prompts[0][0]
    assert "say(<message>)" not in clients[2].prompts[0][0]
    assert "Communication is disabled" in clients[2].prompts[0][1]
    assert all(s["invalid_actions"] == s["actions"] for s in records[1]["metrics"]["per_agent"].values())
    summaries = json.loads((folder / "summary.json").read_text())
    assert len(summaries) == 2
    assert {s["communication"] for s in summaries} == {"enabled", "disabled"}
    assert all(s["mean_llm_actions"] == 10 for s in summaries)
    assert sum(s["llm_actions"] for s in summaries) == 40
    for r in records:
        replay(r["path"])
    pairs = json.loads((folder / "communication_comparison.json").read_text())
    assert pairs["completed_matched_pairs"] == 1
    # A changed model setting must invalidate, rather than silently pool, a pair.
    path = Path(records[1]["path"])
    events = read_events(path)
    events[0]["metadata"]["temperature"] = 0.9
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    summarize_batch(folder)
    pairs = json.loads((folder / "communication_comparison.json").read_text())
    assert pairs["completed_matched_pairs"] == 0 and pairs["pairs"][0]["status"] == "settings_mismatch"


def test_partial_pair_stays_visible(tmp_path):
    cfg = config()
    plan = make_plan(cfg, load_suite("rooms/diagnostics/manifest.json"), offline=True)[:1]
    execute_plan(cfg, plan, tmp_path / "partial", offline=True)
    pairs = json.loads((tmp_path / "partial/communication_comparison.json").read_text())
    assert pairs["completed_matched_pairs"] == 0
    assert pairs["pairs"][0]["status"] == "missing_condition"


def test_public_actions_cannot_bypass_disabled_channels():
    with pytest.raises(ValueError, match="private partner actions"):
        MotiveRoomsEnv(generate_room("private_handoffs", 0).room,
                       communication="disabled", partner_actions="public")
    with pytest.raises(ValueError, match="communication mode"):
        LLMAgentPolicy("A", None, communication="typo")


def test_explained_instructions_describe_optional_help_and_match_channel_state():
    original = LLMAgentPolicy("A", None).system_prompt()
    assert "COMMUNICATION FOR THIS ATTEMPT" not in original
    explained = LLMAgentPolicy("A", None, communication_instruction="explained").system_prompt()
    assert "provide a shorter route and save turns" in explained
    assert "not every room benefits" in explained
    assert "on the next step" in explained and "Sending a message costs one turn" in explained
    assert "Communication is optional" in explained
    assert "Once you escape or run out of turns" in explained
    disabled = LLMAgentPolicy("A", None, communication="disabled",
                              communication_instruction="explained").system_prompt()
    assert "communication is OFF" in disabled
    assert "You can communicate directly" not in disabled
    assert "say(<message>)" not in disabled
    cfg = yaml.safe_load(Path("config/eval_collaboration_explained.yaml").read_text())
    plan = make_plan(cfg, load_suite("rooms/diagnostics/manifest.json"))
    assert len(plan) == 12
    assert all(p[1]["communication_instruction"] == "explained" for p in plan)
    assert {p[1]["communication"] for p in plan} == {"enabled", "disabled"}
    with pytest.raises(ValueError, match="communication instruction"):
        make_plan(dict(cfg, communication_instruction="typo"), [])
