"""Exploration dependencies, private discoveries and report-only messaging."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import yaml

from motiverooms.actions import ActionParseError, parse_action
from motiverooms.connected_generator import ConnectedRoomsConfig
from motiverooms.env import MotiveRoomsEnv
from motiverooms.experiment import replay, run_episode
from motiverooms.exploration_generator import generate_exploration_episode
from motiverooms.llm_policy import LLMAgentPolicy
from motiverooms.orchestration import ScriptedOrchestrator, reference_for_team, run_orchestrated_episode
from motiverooms.schema import Room, RoomValidationError
from motiverooms.tool_calls import ToolCall, ToolResponse
from scripts.evaluate import execute_plan, load_cases, make_plan


@pytest.mark.parametrize("count", [1, 2, 3, 5, 10])
def test_seeded_exploration_maps_have_executable_references(count):
    for seed in range(8):
        case = generate_exploration_episode(seed, ConnectedRoomsConfig(count), version=1)
        assert len(case.room.locations) == count
        assert all(p.requires is None for p in case.room.passages)
        assert len(case.room.codes["exit_code"]) == 2 * max(3, count)
        fragments = dict(node.text.split(": ") for node in case.room.nodes
                         if node.kind == "clue" and node.id != "wall_plaque")
        order = case.room.node("wall_plaque").text.split(": ")[1].split(". Preserve")[0].split(" → ")
        assert "".join(fragments[symbol] for symbol in order) == case.room.codes["exit_code"]
        assert Room.from_dict(case.room.to_dict()) == case.room
        env = MotiveRoomsEnv(case.room, turn_budget=case.metadata["recommended_turn_budget"])
        for tick in case.metadata["reference"]:
            outcomes = env.step_joint(tick)
            assert all(o.valid and o.result != "wrong" for o in outcomes.values())
        assert env.round_over and all(s.escaped for s in env.engine.agents.values())


def test_seed_changes_placements_fragments_and_order_without_global_randomness():
    import random
    state = random.getstate()
    cases = [generate_exploration_episode(seed, version=1) for seed in range(10)]
    assert random.getstate() == state
    assert generate_exploration_episode(42, version=1) == generate_exploration_episode(42, version=1)
    assert len({c.room.codes["exit_code"] for c in cases}) > 1
    assert len({c.room.node("painting").location for c in cases}) > 1
    assert len({c.room.node("wall_plaque").text for c in cases}) > 1


def test_search_item_dependencies_and_hidden_answers():
    case = generate_exploration_episode(42, version=1)
    env = MotiveRoomsEnv(case.room, turn_budget=60, coordination=True, worker_broadcast=False)
    engine = env.engine
    initial = env.observe("A")
    assert "6-digit" in initial
    assert case.room.codes["exit_code"] not in initial
    assert "unpowered_torch (type=" not in initial
    assert "desk_key (item; source=doormat)" not in initial
    assert not engine.step("A", "take(desk_key)").valid
    assert engine.step("A", "inspect(doormat)").valid
    assert "desk_key (item; source=doormat)" in env.observe("A")
    assert engine.step("A", "take(desk_key)").valid
    assert engine.step("A", "use(desk_key, desk_drawer)").valid
    assert "unpowered_torch (type=container)" in env.observe("A")
    assert not engine.step("A", "take(uv_torch)").valid
    assert not engine.step("A", "use(batteries, unpowered_torch)").valid
    storage = case.room.node("tool_cabinet").location
    lounge = case.room.node("painting").location
    assert engine.step("A", f"move({storage})").valid  # passages open initially
    assert not engine.step("A", "take(batteries)").valid
    engine.step("A", "inspect(tool_cabinet)")
    engine.step("A", "take(batteries)")
    assert engine.step("A", "move(room_1)").valid
    assert engine.step("A", "use(batteries, unpowered_torch)").valid
    assert engine.step("A", "take(uv_torch)").valid
    engine.step("A", f"move({lounge})")
    assert "painting_inscription (type=" not in env.observe("A")
    assert not engine.step("A", "inspect(painting_inscription)").valid
    assert engine.step("A", "use(uv_torch, painting)").valid
    assert "painting_inscription (type=clue)" in env.observe("A")
    assert "painting_inscription" not in engine.agents["A"].read_codes
    result = engine.step("A", "inspect(painting_inscription)")
    assert result.valid and "moon:" in result.result
    assert engine.agents["A"].read_codes["painting_inscription"] == case.room.node("painting_inscription").text
    assert "painting_inscription" not in engine.agents["B"].read_codes
    assert case.room.node("painting_inscription").text not in env.observe("B")


@pytest.mark.parametrize("interface", ["text", "tools"])
def test_broadcast_off_tool_contract_and_prompts(interface):
    policy = LLMAgentPolicy("worker_1", None, navigation=True, coordination=True,
                            teammates=("worker_1", "worker_2", "worker_3"),
                            worker_broadcast=False, controller_observes_chat=True,
                            action_interface=interface, communication_instruction="explained", coaching=True)
    names = {t["name"] for t in policy.tools()}
    assert "say" not in names
    assert {"report", "write_note", "read_notes", "give_to"} <= names
    prompt = policy.system_prompt()
    assert "say(" not in prompt
    assert "Public say messages" not in prompt
    assert "report(message)" in prompt
    assert "communicate directly with the other agent" not in prompt
    with pytest.raises(ActionParseError, match="Unavailable"):
        ToolCall("one", "say", '{"message":"hidden broadcast"}').action(worker_broadcast=False)


def test_engine_blocks_text_broadcast_but_keeps_reports_notes_and_transfers(tmp_path):
    case = generate_exploration_episode(42, version=1)
    env = MotiveRoomsEnv(case.room, turn_budget=16, coordination=True, worker_broadcast=False)
    plans = {
        "A": ["say(blocked)", "report(private discovery)", "write_note(star: 19)",
              "inspect(doormat)", "take(desk_key)", "give(desk_key)"],
        "B": ["wait()", "wait()", "wait()", "read_notes()", "wait()", "wait()"],
    }

    class Policy:
        def __init__(self, agent):
            self.agent = agent

        def choose_action(self, observation):
            if self.agent == "B":
                assert "private discovery" not in observation
                if env.env_step < 4:
                    assert "star: 19" not in observation
            line = plans[self.agent][env.env_step]
            return line, line

    result = run_episode(env, {a: Policy(a) for a in env.agent_ids}, path=tmp_path / "channels.jsonl", max_steps=6)
    assert not env.engine.messages
    assert env.engine.reports[0].text == "private discovery"
    assert env.engine.notes == ["star: 19"]
    assert "desk_key" in env.engine.agents["B"].bag
    stats = result["metrics"]["per_agent"]
    assert stats["A"]["invalid_actions"] == 1
    assert stats["A"]["messages_sent"] == 0
    assert stats["A"]["reports_sent"] == stats["A"]["notes_written"] == stats["A"]["items_transferred"] == 1
    assert stats["B"]["notes_read"] == 1
    assert replay(tmp_path / "channels.jsonl") == result["metrics"]


@pytest.mark.parametrize("version", [1, 2])
def test_native_tools_orchestrator_complete_exploration_and_replay(tmp_path, version):
    case = generate_exploration_episode(42, version=version)
    requests = []

    class Client:
        def __init__(self, agent, env):
            self.agent, self.env = agent, env

        def complete_tools(self, **kwargs):
            requests.append(deepcopy(kwargs))
            tick = reference_for_team(case.metadata, self.env.agent_ids)["reference"][self.env.env_step]
            action = parse_action(tick[self.agent])
            schema = next(t for t in kwargs["tools"] if t["name"] == action.verb)
            return ToolResponse((dict(type="function_call", name=action.verb,
                                     call_id=f"{self.agent}_{self.env.env_step}",
                                     arguments=json.dumps(dict(zip(schema["parameters"]["required"], action.args)))),))

    def factory(agent, env, assignment):
        return LLMAgentPolicy(agent, Client(agent, env), shared_room=True, navigation=True,
                              coordination=True, teammates=env.agent_ids, assignment=assignment,
                              worker_broadcast=env.worker_broadcast, history_turns=3)

    controller = ScriptedOrchestrator(context_version="v2", history_turns=3)
    path = tmp_path / "native.jsonl"
    result = run_orchestrated_episode(case.room, controller, factory, path=path, metadata=case.metadata,
                                      turn_budget=case.metadata["recommended_turn_budget"], worker_broadcast=False)
    assert result["end"]["reason"] == "success"
    assert replay(path) == result["metrics"]
    assert result["events"][0]["config"]["worker_broadcast"] is False
    assert requests and all("say" not in {t["name"] for t in r["tools"]} for r in requests)
    assert controller.events[0]["input"]["worker_broadcast"] is False
    assert all(case.room.codes["exit_code"] not in json.dumps(e["input"]) for e in controller.events)


def test_new_evaluation_config_offline_and_broadcast_conditions_do_not_pool(tmp_path):
    config = yaml.safe_load(open("config/eval_exploration_orchestrated.yaml"))
    config["progress"] = False
    cases = load_cases(config, "unused")
    plan = make_plan(config, cases, offline=True)
    plan += make_plan(dict(config, worker_broadcast=True), cases, offline=True)
    records = execute_plan(config, plan, tmp_path / "batch", offline=True)
    assert all(r["metrics"]["joint_escape"] and r["metrics"]["api_calls"] == 0 for r in records)
    summary = json.loads((tmp_path / "batch/summary.json").read_text())
    assert len(summary) == 2 and {s["worker_broadcast"] for s in summary} == {False, True}
    assert "Communication channels" in (tmp_path / "batch/summary.md").read_text()
    with pytest.raises(ValueError, match="orchestrated"):
        make_plan(dict(config, team_mode="fixed"), cases)
    with pytest.raises(ValueError, match="boolean"):
        make_plan(dict(config, worker_broadcast="false"), cases)


def test_schema_rejects_ambiguous_clues_and_unknown_search_contents():
    case = generate_exploration_episode(42, version=1)
    with pytest.raises(RoomValidationError, match="exactly one"):
        replace(case.room, nodes=tuple(replace(n, code="exit_code") if n.id == "wall_plaque" else n for n in case.room.nodes))
    with pytest.raises(RoomValidationError, match="unknown id"):
        replace(case.room, nodes=tuple(replace(n, contents=("missing",)) if n.id == "doormat" else n for n in case.room.nodes))
