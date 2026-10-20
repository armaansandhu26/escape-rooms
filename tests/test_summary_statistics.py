from copy import deepcopy
import json

import pytest

from motiverooms.benchmark_policies import ScriptedPolicy
from motiverooms.env import MotiveRoomsEnv
from motiverooms.evaluation import summarize
from motiverooms.experiment import run_episode
from motiverooms.generator import generate_room


def test_step_averages_include_failures_and_preserve_individual_escape_timing(tmp_path):
    case = generate_room("private_handoffs", 0)
    env = MotiveRoomsEnv(case.room, turn_budget=10)
    policies = {"A": ScriptedPolicy("A", env, case.metadata, "honest"),
                "B": ScriptedPolicy("B", env, case.metadata, "silent")}
    result = run_episode(env, policies, metadata=case.metadata)
    metrics = result["metrics"]
    assert metrics["joint_steps"] == 8
    assert metrics["per_agent"]["B"]["escape_step"] == 6
    meta = dict(case.metadata, source_room_id=case.room.id, condition="honest", evaluated_seat="B",
                repeat=0, mode="mixed", observation_version="v3")
    records = [dict(metadata=meta, metrics=metrics, path="success.jsonl")]
    failed = deepcopy(metrics)
    failed.update(termination_reason="turn_budget", joint_escape=False, joint_steps=10)
    failed["per_agent"]["B"].update(escaped=False, actions=10, escape_step=None, payoff=0)
    records.append(dict(metadata=meta, metrics=failed, path="failure.jsonl"))
    summary = summarize(records, tmp_path)[0]
    assert summary["mean_completed_joint_steps"] == summary["median_completed_joint_steps"] == 9
    assert summary["mean_successful_joint_steps"] == 8
    assert summary["std_completed_joint_steps"] == pytest.approx(2 ** .5)
    assert summary["mean_llm_actions"] == 8
    assert summary["mean_llm_escape_actions"] == summary["mean_llm_escape_step"] == 6
    assert summary["llm_escape_rate"] == .5
    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "condition_summary.csv").is_file()
    assert json.loads((tmp_path / "condition_summary.json").read_text())[0]["mean_completed_joint_steps"] == 9
    assert "Avg joint steps" in (tmp_path / "summary.md").read_text()


def test_no_sharing_opportunities_stays_undefined_and_errors_are_excluded_from_step_mean(tmp_path):
    case = generate_room("private_handoffs", 4, profile="zero")
    env = MotiveRoomsEnv(case.room)
    policies = {a: ScriptedPolicy(a, env, case.metadata, "silent") for a in env.agent_ids}
    result = run_episode(env, policies, metadata=case.metadata)
    meta = dict(case.metadata, source_room_id=case.room.id, condition="silent", evaluated_seat="A",
                repeat=0, mode="mixed", observation_version="v3")
    records = [dict(metadata=meta, metrics=result["metrics"], path="success.jsonl")]
    errored = deepcopy(result["metrics"])
    errored.update(termination_reason="api_error", joint_steps=100)
    records.append(dict(metadata=meta, metrics=errored, path="error.jsonl"))
    summary = summarize(records, tmp_path)[0]
    assert summary["llm_sharing_opportunities"] == 0
    assert summary["llm_sharing_rate"] is None
    assert summary["mean_completed_joint_steps"] == 2
    assert summary["error_or_truncation_counts"] == {"api_error": 1}
