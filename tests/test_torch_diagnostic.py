"""Torch affordances, isolated native-tool runs, and controlled budget retries."""
from dataclasses import replace
import json
from pathlib import Path
import random

import pytest
import yaml

from motiverooms.actions import parse_action
from motiverooms.azure_openai import AzureOpenAISettings
from motiverooms.env import MotiveRoomsEnv
from motiverooms.experiment import replay
from motiverooms.exploration_generator import generate_exploration_episode
from motiverooms.tool_calls import ToolResponse
from motiverooms.torch_puzzle import generate_torch_episode
from scripts.evaluate import execute_plan, load_cases, make_plan
from scripts.inspect_progress import inspect_log


def config(name):
    with open(f"config/{name}.yaml") as stream:
        return dict(yaml.safe_load(stream), progress=False)


def test_station_clarification_changes_only_descriptions_and_protocol_identity():
    legacy = generate_exploration_episode(42, version=1)
    explicit = generate_exploration_episode(42, version=1, torch_affordance="explicit_station")
    assert legacy.metadata["reference"] == explicit.metadata["reference"]
    assert legacy.room == replace(explicit.room, id=legacy.room.id, nodes=legacy.room.nodes)
    for old, new in zip(legacy.room.nodes, explicit.room.nodes):
        assert old == replace(new, desc=old.desc)
    assert "cannot be picked up" in explicit.room.node("unpowered_torch").desc
    assert explicit.metadata["torch_affordance"] == "explicit_station"
    assert "fixed torch assembly station" in generate_exploration_episode(42).room.node("unpowered_torch").desc


def test_torch_diagnostic_has_real_prerequisites_and_no_initial_answer():
    case = generate_torch_episode(0)
    env = MotiveRoomsEnv(case.room, turn_budget=12)
    observation = env.observe("A")
    assert env.agent_ids == ("A",)
    assert case.room.codes["exit_code"] not in observation
    assert "cannot be picked up" in observation
    assert not env.engine.step("A", "take(unpowered_torch)").valid
    assert not env.engine.step("A", "inspect(painting_inscription)").valid
    assert not env.engine.step("A", "use(batteries, unpowered_torch)").valid
    assert not env.engine.step("A", "take(uv_torch)").valid
    for action in case.metadata["reference_actions"]:
        result = env.engine.step("A", action)
        assert result.valid and result.result != "wrong"
    assert env.engine.agents["A"].escaped


def test_torch_seeds_are_reproducible_and_do_not_change_global_rng():
    state = random.getstate()
    cases = [generate_torch_episode(seed) for seed in range(10)]
    assert random.getstate() == state
    assert generate_torch_episode(0) == cases[0]
    assert len({case.room.codes["exit_code"] for case in cases}) > 1
    assert {n.id for case in cases for n in case.room.nodes if n.kind == "spot"} == {"shelf", "workbench"}


def test_all_torch_seeds_complete_offline_and_replay_with_measured_milestones(tmp_path):
    settings = config("eval_torch_dependency")
    plan = make_plan(settings, load_cases(settings, "unused"), offline=True)
    records = execute_plan(settings, plan, tmp_path / "offline", offline=True)
    assert len(records) == 10
    for record in records:
        assert record["metrics"]["joint_escape"] and record["metrics"]["joint_steps"] == 6
        assert record["metrics"]["api_calls"] == 0
        assert replay(record["path"]) == record["metrics"]
        progress = inspect_log(Path(record["path"]))
        assert dict(progress["milestones"])["Torch powered"] == 2
        assert dict(progress["milestones"])["Painting clue read"] == 5
        assert dict(progress["milestones"])["Escaped"] == 6
        assert progress["repeated_inspections"] == progress["invalid_actions"] == 0


def test_solo_diagnostic_uses_native_calls_without_controller_or_answer_leak(tmp_path, monkeypatch):
    settings = config("eval_torch_dependency")
    settings["seeds"] = [0]
    case = generate_torch_episode(0)
    calls = []
    model = AzureOpenAISettings("test", "https://example.com/openai/v1/", "test")
    monkeypatch.setattr(AzureOpenAISettings, "from_env", lambda: model)

    class Client:
        def complete_tools(self, **request):
            index = len(calls)
            calls.append(request)
            assert "spawn_subagents" not in {tool["name"] for tool in request["tools"]}
            if index == 0:
                assert case.room.codes["exit_code"] not in json.dumps(request["input_items"])
            action = parse_action(case.metadata["reference_actions"][index])
            schema = next(tool for tool in request["tools"] if tool["name"] == action.verb)
            return ToolResponse((dict(type="function_call", name=action.verb, call_id=f"c{index}",
                                      arguments=json.dumps(dict(zip(schema["parameters"]["required"], action.args)))),))

    plan = make_plan(settings, load_cases(settings, "unused"))
    records = execute_plan(settings, plan, tmp_path / "native", client_factory=lambda *_args, **_kwargs: Client())
    assert len(calls) == 6 and records[0]["metrics"]["joint_escape"]
    assert len(records[0]["metrics"]["per_agent"]) == 1
    assert replay(records[0]["path"]) == records[0]["metrics"]


def test_original_and_station_retries_have_same_mechanics_and_60_executed_turns(tmp_path):
    original = config("eval_exploration_retry_60")
    station = config("eval_exploration_retry_60_station")
    assert {key for key in original if original[key] != station[key]} == {"experiment", "torch_affordance"}
    cases = load_cases(original, "unused") + load_cases(station, "unused")
    plan = make_plan(original, cases, offline=True)
    records = execute_plan(original, plan, tmp_path / "retry", offline=True)
    for record in records:
        assert record["metadata"]["turn_budget"] == record["metadata"]["max_steps"] == 60
        assert record["metrics"]["joint_escape"]
        assert record["metrics"]["joint_steps"] == 20
        assert inspect_log(Path(record["path"]))["turn_budget"] == 60
        assert replay(record["path"]) == record["metrics"]
    summaries = json.loads((tmp_path / "retry/summary.json").read_text())
    assert len(summaries) == 2
    assert {row["torch_affordance"] for row in summaries} == {"legacy", "explicit_station"}


def test_step_cutoff_is_independent_of_per_agent_turn_budget(tmp_path):
    settings = config("eval_exploration_retry_60")
    settings["max_steps"] = 7
    plan = make_plan(settings, load_cases(settings, "unused"), offline=True)
    record, = execute_plan(settings, plan, tmp_path / "cutoff", offline=True)
    assert record["metrics"]["termination_reason"] == "max_steps"
    assert record["metrics"]["joint_steps"] == 7
    assert record["metadata"]["turn_budget"] == 60


@pytest.mark.parametrize("option", ["turn_budget", "max_steps"])
@pytest.mark.parametrize("value", [0, -1, True, "60"])
def test_invalid_budget_overrides_fail_before_execution(option, value):
    with pytest.raises(ValueError, match="positive integer"):
        make_plan({option: value}, [])


def test_invalid_torch_configurations_fail_early():
    with pytest.raises(ValueError, match="torch_affordance"):
        generate_exploration_episode(42, torch_affordance="automatic")
    settings = config("eval_torch_dependency")
    with pytest.raises(ValueError, match="num_rooms"):
        load_cases(dict(settings, num_rooms=2), "unused")
    with pytest.raises(ValueError, match="team_mode"):
        load_cases(dict(settings, team_mode="orchestrated"), "unused")
