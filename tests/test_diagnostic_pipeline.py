from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from motiverooms.azure_openai import AzureOpenAISettings, AzureResponsesClient
from motiverooms.benchmark_policies import ScriptedPolicy
from motiverooms.env import MotiveRoomsEnv
from motiverooms.evaluation import load_suite, solo_projection, summarize
from motiverooms.experiment import compute_metrics, replay, run_episode
from motiverooms.generator import FAMILIES, generate_room
from motiverooms.schema import load_room
from motiverooms.telemetry import BudgetExceeded, RequestBudget
from scripts.evaluate import execute_plan, make_plan
from scripts.gen_rooms import build_suite, validate_case


@pytest.mark.parametrize("family", FAMILIES)
def test_generated_graphs_are_seeded_solo_solvable_and_have_real_help_value(family):
    case = generate_room(family, 0)
    assert case == generate_room(family, 0)
    assert case.room != generate_room(family, 1).room
    ticks = validate_case(case)
    heldout = generate_room(family, 0, split="heldout")
    validate_case(heldout)
    assert len(heldout.room.nodes) > len(case.room.nodes)
    for seat in ("A", "B"):
        projection = solo_projection(case.room, seat)
        assert set(projection.wings) == {"A"}
    if family == "parallel_branches":
        assert ticks < case.metadata["oracle"]["A"]["solo_turns"]
    else:
        assert all(s["help_value"] > 0 for s in case.metadata["oracle"].values())
        zero = generate_room(family, 4, profile="zero")
        assert not zero.metadata["handoffs"]
        assert all(s["help_value"] == 0 for s in zero.metadata["oracle"].values())
        asym = generate_room(family, 3, profile="asymmetric")
        assert asym.metadata["oracle"]["A"]["help_value"] > asym.metadata["oracle"]["B"]["help_value"]


def test_frozen_suite_detects_modification_before_overwriting(tmp_path):
    build_suite(tmp_path, count=1)
    manifest = tmp_path / "manifest.json"
    original = manifest.read_bytes()
    path = next(tmp_path.rglob("*.yaml"))
    path.write_text(path.read_text().replace("Code for", "Changed code for"))
    changed = path.read_bytes()
    with pytest.raises(ValueError, match="hash mismatch"):
        load_suite(manifest)
    with pytest.raises(ValueError, match="Existing room differs"):
        build_suite(tmp_path, count=1)
    assert path.read_bytes() == changed
    assert manifest.read_bytes() == original


@pytest.mark.parametrize("family", FAMILIES)
def test_reference_logs_replay_and_measure_task_relevant_sharing(tmp_path, family):
    case = generate_room(family, 0)
    env = MotiveRoomsEnv(case.room, turn_budget=case.metadata["recommended_turn_budget"])
    policies = {a: ScriptedPolicy(a, env, case.metadata, "reference") for a in env.agent_ids}
    path = tmp_path / "episode.jsonl"
    result = run_episode(env, policies, path=path, metadata=case.metadata)
    assert result["end"]["reason"] == "success"
    assert replay(path) == result["metrics"]
    assert result["metrics"]["api_calls"] == 0
    stats = result["metrics"]["per_agent"]
    if family != "parallel_branches":
        assert all(s["useful_shares"] == 1 for s in stats.values())
        assert all(s["used_shared_resources"] == 1 for s in stats.values())
        assert all(s["sharing_rate"] == 1 for s in stats.values())
        assert all(s["false_code_messages"] == 0 for s in stats.values())
    lines = path.read_text().splitlines()
    event = next(json.loads(line) for line in lines if json.loads(line)["type"] == "joint_step")
    event["actions"]["A"] = "wait()"
    lines[next(i for i, line in enumerate(lines) if json.loads(line)["type"] == "joint_step")] = json.dumps(event)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="Replay mismatch"):
        replay(path)


def fake_azure(response=None, error=None, budget=None):
    settings = AzureOpenAISettings("secret-not-to-log", "https://example.com/openai/v1/", "test")
    client = object.__new__(AzureResponsesClient)
    client.settings, client.budget, client.calls = settings, budget, []

    def create(**kwargs):
        if error:
            raise error
        return response

    client._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return client


def test_azure_records_usage_latency_and_configured_price_without_secrets():
    budget = RequestBudget(input_usd_per_million=1, output_usd_per_million=2)
    response = SimpleNamespace(output_text="  wait()\n", id="response-1", status="completed",
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20, total_tokens=120))
    client = fake_azure(response=response, budget=budget)
    assert client.complete(instructions="system", user_input="observation") == "wait()"
    record = client.calls[0]
    assert record["raw_text"] == "  wait()\n"
    assert record["estimated_cost_usd"] == pytest.approx(.00014)
    assert record["latency_seconds"] >= 0
    assert record["usage"]["total_tokens"] == budget.tokens == 120
    assert "secret-not-to-log" not in json.dumps(record)


def test_budget_reservations_block_parallel_overspending_and_charge_unknown_requests():
    budget = RequestBudget(max_calls=2, max_tokens=1000)
    reservation = budget.reserve("", "", 300)
    with pytest.raises(BudgetExceeded, match="Token"):
        budget.reserve("", "", 300)
    budget.settle(reservation, None)
    assert budget.tokens == 556
    with pytest.raises(BudgetExceeded):
        budget.reserve("", "", 300)
    with pytest.raises(ValueError, match="both token prices"):
        RequestBudget(max_cost_usd=1)


@pytest.mark.parametrize("failure,expected", [(TimeoutError("do not log secrets"), "api_error"),
                                               (BudgetExceeded("cap"), "budget_exceeded")])
def test_failed_requests_produce_replayable_partial_logs(tmp_path, failure, expected):
    case = generate_room("private_handoffs", 0)
    env = MotiveRoomsEnv(case.room)
    client = fake_azure(error=failure, budget=RequestBudget())

    class Policy:
        def __init__(self):
            self.client = client

        def choose_action(self, observation):
            raw = self.client.complete(instructions="system", user_input=observation)
            return raw, raw

    path = tmp_path / "failed.jsonl"
    policies = {"A": Policy(), "B": ScriptedPolicy("B", env, case.metadata, "idle")}
    result = run_episode(env, policies, path=path)
    assert result["end"]["reason"] == expected
    assert result["metrics"]["joint_steps"] == 0
    assert replay(path)["termination_reason"] == expected
    assert "do not log secrets" not in path.read_text()


def test_batch_plan_solo_roles_and_offline_execution(tmp_path):
    manifest = Path("rooms/diagnostics/manifest.json")
    cases = load_suite(manifest)
    config = dict(split="development", seeds=[0], both_seats=True, repeats=1,
                  conditions=["solo", "honest", "silent", "deceptive", "self_play"], mode="mixed")
    plan = make_plan(config, cases)
    assert len(plan) == 29
    assert {p[3] for p in plan if p[2] == "solo" and not p[0].is_shared} == {"A", "B"}
    offline = make_plan(config, cases, offline=True)
    records = execute_plan(config, offline, tmp_path / "offline", offline=True)
    assert len(records) == 3
    assert all(r["metrics"]["joint_escape"] for r in records)
    assert (tmp_path / "offline/episodes.csv").is_file()
    assert (tmp_path / "offline/summary.md").is_file()


def test_summary_keeps_api_failures_out_of_task_denominator(tmp_path):
    meta = dict(family="x", profile="balanced", split="development", source_room_id="room",
                seed=0, condition="self_play", repeat=0, evaluated_seat="A", mode="mixed")
    metrics = dict(termination_reason="api_error", joint_escape=False, escape_rate=0, joint_steps=0,
                   total_actions=0, api_calls=1, total_tokens=0, unknown_usage_calls=1,
                   estimated_cost_usd=None, per_agent={})
    rows = [dict(metadata=meta, metrics=metrics, path="failed")]
    completed = deepcopy(metrics)
    completed.update(termination_reason="success", joint_escape=True, escape_rate=1)
    rows.append(dict(metadata=meta, metrics=completed, path="completed"))
    summary = summarize(rows, tmp_path)[0]
    assert summary["completed"] == 1
    assert summary["failures_or_truncations"] == 1
    assert summary["completed_joint_escape_rate"] == 1


def test_all_baseline_conditions_execute_with_mock_azure_and_replay(tmp_path, monkeypatch):
    config = dict(split="development", seeds=[0], both_seats=True, repeats=1,
                  conditions=["solo", "honest", "silent", "deceptive", "self_play"],
                  mode="mixed", budget=dict(max_calls=10000, max_tokens=10_000_000))
    plan = make_plan(config, load_suite("rooms/diagnostics/manifest.json"))

    def factory(settings, budget):
        response = SimpleNamespace(output_text="wait()", id="mock", status="completed",
                                   usage=dict(input_tokens=10, output_tokens=2, total_tokens=12))
        return fake_azure(response=response, budget=budget)

    import scripts.evaluate as evaluation_script
    # No credentials or network are needed for mocked baseline execution.
    class Settings:
        @staticmethod
        def from_env():
            return AzureOpenAISettings("test", "https://example.com/openai/v1/", "test")
    monkeypatch.setattr(evaluation_script, "AzureOpenAISettings", Settings)
    records = execute_plan(config, plan, tmp_path / "mock_matrix", client_factory=factory)
    assert len(records) == len(plan)
    assert all(r["metrics"]["termination_reason"] in {"success", "turn_budget"} for r in records)
    assert all(r["metrics"]["unknown_usage_calls"] == 0 for r in records)


def test_empty_provider_text_is_model_failure_not_transport_failure():
    response = SimpleNamespace(output_text="", usage=dict(input_tokens=10, output_tokens=300, total_tokens=310))
    client = fake_azure(response=response)
    assert client.complete(instructions="system", user_input="observation") == ""
    assert client.calls[0]["error"] is None
    assert client.calls[0]["output_error"] == "empty_text"


def test_descriptive_report_rebuild_preserves_evidence_and_explains_failures(tmp_path):
    from scripts.summarize_runs import summarize_batch

    case = generate_room("parallel_branches", 0)
    env = MotiveRoomsEnv(solo_projection(case.room, "A"), turn_budget=3)

    class InvalidPolicy:
        def choose_action(self, observation):
            return "take(not_an_item)", "take(not_an_item)"

    metadata = dict(case.metadata, source_room_id=case.room.id, evaluated_seat="A",
                    condition="solo", repeat=0, mode="mixed", model="test")
    path = tmp_path / "episode.jsonl"
    result = run_episode(env, {"A": InvalidPolicy()}, path=path, metadata=metadata)
    stats = result["metrics"]["per_agent"]["A"]
    assert stats["longest_failed_action_streak"] == 3
    assert stats["new_locks_opened"] == 0
    original = path.read_bytes()
    report = summarize_batch(tmp_path).read_text()
    assert path.read_bytes() == original
    assert "LLM + helpful script" in report
    assert "LLM + solo-route script" in report
    assert "no RL training" in report
    assert "3 consecutive turns" in report
    assert "3/3" in report
    assert "one room instance" in report
