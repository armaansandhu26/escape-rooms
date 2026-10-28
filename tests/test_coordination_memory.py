"""Persistent report provenance and worker-owned progress without oracle access."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from motiverooms.actions import ActionParseError
from motiverooms.coordination_memory import CoordinationMemory, empty_updates, memory_update_schema
from motiverooms.engine import ActionOutcome
from motiverooms.env import MotiveRoomsEnv
from motiverooms.experiment import replay
from motiverooms.exploration_generator import generate_exploration_episode
from motiverooms.llm_policy import LLMAgentPolicy
from motiverooms.memory import ObservedActionMemory
from motiverooms.orchestration import Orchestrator, reference_for_team, run_orchestrated_episode, team_room
from motiverooms.tool_calls import ToolResponse
from scripts.evaluate import execute_plan, load_cases, make_plan


AGENTS = ("worker_1", "worker_2")


def native(name, arguments):
    return ToolResponse((dict(type="function_call", name=name, arguments=json.dumps(arguments), call_id="one"),))


def reported_memory():
    memory = CoordinationMemory()
    memory.add_evidence("report_1", agent_id="worker_1", message="I have batteries in room_3.", channel="report", step=2)
    memory.add_evidence("report_2", agent_id="worker_2", message="Mechanism in room_1 needs batteries.", channel="report", step=3)
    return memory


def battery_updates():
    changes = empty_updates()
    changes["items"] = [dict(item_id="batteries", holder="worker_1", room_id="room_3", status="held", evidence_ids=["report_1"])]
    changes["dependencies"] = [dict(target_id="mechanism", needed_item_id="batteries", status="unresolved", evidence_ids=["report_2"])]
    changes["searches"] = [dict(node_id="mechanism", room_id="room_1", status="blocked", evidence_ids=["report_2"])]
    changes["locations"] = [dict(agent_id="worker_1", room_id="room_3", evidence_ids=["report_1"])]
    return changes


def test_board_persists_blocked_dependencies_holders_and_original_sources():
    memory = reported_memory()
    changes = battery_updates()
    memory.apply(changes, AGENTS, step=3)
    changes["items"][0]["holder"] = "worker_2"
    before = memory.view()
    memory.apply(empty_updates(), AGENTS, step=20)
    assert memory.view() == before
    assert before["items"][0]["holder"] == "worker_1"
    assert before["dependencies"][0]["status"] == "unresolved"
    assert before["searches"][0]["status"] == "blocked"
    assert all(row["verified"] is False for entries in before.values() for row in entries)
    for i in range(3, 30):
        memory.add_evidence(f"report_{i}", agent_id="worker_2", message="No change.", channel="report", step=i)
    sources = {r["evidence_id"]: r for r in memory.supporting_evidence()}
    assert sources["report_1"]["message"] == "I have batteries in room_3."
    assert sources["report_2"]["agent_id"] == "worker_2"
    before["dependencies"].clear()
    assert memory.view()["dependencies"]  # defensive copies


@pytest.mark.parametrize("mutate", [
    lambda u: u["items"][0].update(evidence_ids=["invented"]),
    lambda u: u["items"][0].update(evidence_ids=[]),
    lambda u: u["items"][0].update(holder="stranger"),
    lambda u: u["items"][0].update(holder=None),
    lambda u: u["searches"][0].update(status="magic"),
    lambda u: u["dependencies"][0].update(target_id="bad target"),
    lambda u: u["locations"][0].update(agent_id="stranger"),
    lambda u: u["searches"].append(dict(u["searches"][0])),
    lambda u: u["searches"][0].update(extra="unsupported"),
])
def test_invalid_delta_is_rejected_atomically(mutate):
    memory = reported_memory()
    memory.apply(battery_updates(), AGENTS, step=3)
    original = memory.view()
    changes = battery_updates()
    mutate(changes)
    with pytest.raises(ActionParseError):
        memory.apply(changes, AGENTS, step=3)
    assert memory.view() == original


def test_new_evidence_can_revise_holder_and_resolve_a_dependency():
    memory = reported_memory()
    memory.apply(battery_updates(), AGENTS, step=3)
    memory.add_evidence("report_3", agent_id="worker_1", message="Used batteries on mechanism in room_1; it opened.", channel="report", step=8)
    changes = empty_updates()
    changes["items"] = [dict(item_id="batteries", holder="worker_1", room_id="room_1", status="held", evidence_ids=["report_3"])]
    changes["dependencies"] = [dict(target_id="mechanism", needed_item_id="batteries", status="resolved", evidence_ids=["report_3"])]
    memory.apply(changes, AGENTS, step=8)
    assert len(memory.view()["items"]) == 1
    assert memory.view()["dependencies"][0]["status"] == "resolved"
    assert memory.view()["dependencies"][0]["evidence_ids"] == ["report_3"]


def test_provenance_validation_does_not_claim_to_verify_report_truth():
    memory = CoordinationMemory()
    memory.add_evidence("report_1", agent_id="worker_1", message="I claim to hold batteries.", channel="report", step=1)
    changes = empty_updates()
    changes["items"] = [dict(item_id="batteries", holder="worker_1", room_id=None, status="held", evidence_ids=["report_1"])]
    memory.apply(changes, AGENTS, step=2)
    assert memory.view()["items"][0]["verified"] is False


def test_v3_retains_dependency_after_history_trim_and_keeps_private_discoveries_hidden():
    case = generate_exploration_episode(42, version=1)
    requests = []
    updates = empty_updates()
    updates["dependencies"] = [dict(target_id="unpowered_torch", needed_item_id="batteries", status="unresolved", evidence_ids=["report_1"])]
    updates["searches"] = [dict(node_id="unpowered_torch", room_id="room_1", status="blocked", evidence_ids=["report_1"])]
    updates["items"] = [dict(item_id="batteries", holder="worker_2", room_id="room_3", status="held", evidence_ids=["report_2"])]

    class Client:
        def complete_tools(self, **request):
            requests.append(deepcopy(request))
            name = request["tools"][0]["name"]
            view = json.loads(request["input_items"][-1]["content"])
            if name == "spawn_subagents":
                return native(name, dict(assignments=["Explore and report."] * 2, rationale="Two workers."))
            return native(name, dict(assignments=[], rationale="Maintain board.",
                                     memory_updates=updates if view["step"] == 3 else empty_updates()))

    controller = Orchestrator(Client(), context_version="v3", history_turns=1)
    plan = controller.spawn(case.room, turn_budget=16, mode="collaborative", shared_escape="together", worker_broadcast=False)
    env = MotiveRoomsEnv(team_room(case.room, plan), coordination=True, turn_budget=16, worker_broadcast=False)
    policies = {a: SimpleNamespace(assignment="Explore.") for a in plan.agents}
    env.step_joint(dict(worker_1="move(room_3)", worker_2="report(unpowered_torch in room_1 needs batteries.)"))
    controller.coordinate(env, policies)
    # The holder claim is intentionally false; the controller must not inspect
    # the real inventory to correct it. Provenance is not truth verification.
    env.step_joint(dict(worker_1="inspect(shipping_receipt)", worker_2="report(I have batteries in room_3.)"))
    controller.coordinate(env, policies)
    for _ in range(3):
        env.step_joint(dict(worker_1="wait()", worker_2="wait()"))
        controller.coordinate(env, policies)
    view = json.loads(requests[-1]["input_items"][-1]["content"])
    assert view["new_reports"] == []
    assert view["task_board"]["dependencies"][0]["status"] == "unresolved"
    assert view["task_board"]["items"][0]["holder"] == "worker_2"
    assert view["task_board"]["searches"][0]["status"] == "blocked"
    assert "star: 88" not in json.dumps(requests)
    assert case.room.codes["exit_code"] not in json.dumps(requests)
    assert sum(i.get("role") == "user" for i in requests[-1]["input_items"]) == 2
    assert "BLOCKED from EXHAUSTED" in requests[-1]["instructions"]
    assert all("say" not in r["tools"][0]["parameters"]["properties"] for r in requests)


@pytest.mark.parametrize("interface", ["tools", "text"])
def test_worker_memory_keeps_own_failures_counts_and_latest_assignment(interface):
    requests = []

    class Client:
        def complete_tools(self, **request):
            requests.append(request)
            return native("wait", {})

        def complete(self, **request):
            requests.append(request)
            return "wait()"

    from motiverooms.actions import parse_action
    policy = LLMAgentPolicy("worker_1", Client(), coordination=True, progress_memory=True,
                            action_interface=interface, history_turns=1, worker_broadcast=False)
    for action, valid, result in [
        ("inspect(calendar)", True, "No inscriptions."),
        ("inspect(calendar)", True, "No inscriptions."),
        ("take(token)", False, "That item is no longer there."),
    ]:
        policy.record_outcome(ActionOutcome(parse_action(action), valid, result))
    policy.assignment = "Bring batteries to the reported mechanism."
    policy.choose_action("CURRENT ROOM: room_3\nYOUR BAG: batteries")
    content = requests[-1]["input_items"][-1]["content"] if interface == "tools" else requests[-1]["user_input"]
    assert '"attempts": 2' in content and "No inscriptions." in content
    assert '"last_valid": false' in content
    assert "Bring batteries to the reported mechanism." in content
    assert "OWN ACTION PROGRESS" in content
    assert "task_board" not in content  # controller board is not a worker broadcast


def test_progress_memory_is_bounded_and_cannot_be_mutated_through_view():
    memory = ObservedActionMemory(limit=2)
    for i in range(3):
        memory.record(f"inspect(object_{i})", valid=True, result="Empty.")
    view = memory.view()
    assert len(view) == 2 and view[0]["action"] == "inspect(object_1)"
    view[0]["last_result"] = "Changed externally."
    assert memory.view()[0]["last_result"] == "Empty."


def test_worker_progress_counts_malformed_action_turns():
    policy = LLMAgentPolicy("worker_1", None, progress_memory=True)
    policy.record_outcome(ActionOutcome(None, False, "Invalid tool call."))
    assert policy._progress.view()[0]["last_own_turn"] == 1
    assert policy._progress.view()[0]["last_valid"] is False


def test_v3_config_offline_replay_preserves_map_caps_and_separates_protocols(tmp_path):
    old = yaml.safe_load(open("config/eval_exploration_orchestrated.yaml"))
    new = yaml.safe_load(open("config/eval_exploration_orchestrated_v3.yaml"))
    assert old["budget"] == new["budget"]
    assert old["max_subagents"] == new["max_subagents"]
    assert old["model"] == new["model"]
    assert load_cases(old, "unused") == load_cases(new, "unused")
    new["progress"] = False
    plan = make_plan(new, load_cases(new, "unused"), offline=True)
    records = execute_plan(new, plan, tmp_path / "v3", offline=True)
    assert records[0]["metrics"]["joint_escape"] and records[0]["metrics"]["api_calls"] == 0
    assert replay(records[0]["path"]) == records[0]["metrics"]
    summary = json.loads((tmp_path / "v3/summary.json").read_text())
    assert summary[0]["orchestration_protocol"] == "v3" and summary[0]["worker_progress_memory"] is True
    assert "[v3]" in (tmp_path / "v3/summary.md").read_text()
    with pytest.raises(ValueError, match="boolean"):
        make_plan(dict(new, worker_progress_memory="true"), load_cases(new, "unused"))


def test_memory_schema_is_strict_at_every_object_level():
    schema = memory_update_schema(AGENTS)

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    check(schema)


@pytest.mark.parametrize("invalid_source", [False, True])
def test_recorded_bad_memory_response_keeps_assignments_and_recovers_without_retry(tmp_path, invalid_source):
    """Replay the actual failing payload with fake clients; no paid requests."""
    fixture = json.loads((Path(__file__).parent / "fixtures/contradictory_memory_update.json").read_text())
    bad = deepcopy(fixture["arguments"])
    corrected = deepcopy(bad)
    corrected["memory_updates"]["items"][0]["holder"] = None
    corrected["memory_updates"]["dependencies"] = []  # no reported target ID yet
    if invalid_source:
        bad = deepcopy(corrected)
        bad["memory_updates"]["items"][0]["evidence_ids"] = ["invented_source"]
    requests = []

    class Client:
        def complete_tools(self, **request):
            requests.append(deepcopy(request))
            name = request["tools"][0]["name"]
            view = json.loads(request["input_items"][-1]["content"])
            if name == "spawn_subagents":
                return native(name, dict(assignments=["Follow the test reference."] * 3, rationale="Offline regression."))
            if view["step"] == 2:
                return native(name, bad)
            if view["step"] == 3:
                assert view["task_board"] == empty_updates()  # invalid delta was fully rolled back
                outputs = [i for i in request["input_items"] if i.get("type") == "function_call_output"]
                feedback = json.loads(outputs[-1]["output"])
                assert feedback["memory_update_error"]
                assert feedback["updated_assignments"] == {u["agent_id"]: u["task"] for u in bad["assignments"]}
                return native(name, corrected)
            return native(name, dict(assignments=[], rationale="Keep reference.", memory_updates=empty_updates()))

    case = generate_exploration_episode(42, version=1)
    controller = Orchestrator(Client(), context_version="v3", history_turns=1)

    def factory(agent, env, assignment):
        reference = reference_for_team(case.metadata, env.agent_ids)["reference"]
        report = next(r["message"] for r in fixture["reports"] if r["agent_id"] == agent)

        class Worker:
            def __init__(self):
                self.assignment = assignment

            def choose_action(self, observation):
                action = f"report({report})" if env.env_step == 0 else reference[env.env_step - 1][agent]
                return action, action

        return Worker()

    path = tmp_path / "recovered.jsonl"
    result = run_orchestrated_episode(case.room, controller, factory, path=path, metadata=case.metadata,
                                      turn_budget=36, worker_broadcast=False)
    assert result["end"]["reason"] == "success"
    assert replay(path) == result["metrics"]
    assert len(requests) == result["metrics"]["joint_steps"] + 1  # one spawn, one call per tick
    decisions = [r for r in result["events"] if r["type"] == "orchestrator_decision"]
    rejected = [r for r in decisions if "memory_update_error" in (r["result"] or {})]
    assert len(rejected) == 1 and rejected[0]["error"] is None
    assert all(n == 0 for n in rejected[0]["result"]["memory_updates_applied"].values())
    assert controller.memory.view()["items"][0]["status"] == "available"
    assert controller.memory.view()["items"][0]["holder"] is None
