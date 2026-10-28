"""V4 team bounds, report-backed navigation, private help and native replay."""
from copy import deepcopy
import json

import pytest
import yaml

from motiverooms.actions import ActionParseError, parse_action
from motiverooms.connected_generator import generate_connected_episode
from motiverooms.coordination_memory import CoordinationMemory, empty_updates, memory_update_schema
from motiverooms.experiment import replay
from motiverooms.llm_policy import LLMAgentPolicy
from motiverooms.orchestration import Orchestrator, OrchestratorLimits, run_orchestrated_episode, reference_for_team
from motiverooms.tool_calls import ToolResponse
from scripts.evaluate import execute_plan, load_cases, make_plan
from scripts.rollout_llm import parse_args


def native(name, arguments, call_id="call"):
    return ToolResponse((dict(type="function_call", name=name, call_id=call_id,
                              arguments=json.dumps(arguments)),))


class Client:
    def __init__(self, handler):
        self.handler = handler
        self.requests = []

    def complete_tools(self, **request):
        self.requests.append(deepcopy(request))
        return self.handler(request)


@pytest.mark.parametrize("count", [1, 5])
def test_rosters_outside_two_to_four_are_rejected_without_worker_creation(count):
    client = Client(lambda _: native("spawn_subagents", dict(assignments=["Explore."] * count, rationale="Test bounds.")))
    result = run_orchestrated_episode(generate_connected_episode(42).room,
                                      Orchestrator(client, context_version="v4"),
                                      lambda *_: pytest.fail("Invalid roster must not create workers"))
    assert result["end"]["reason"] == "policy_error"
    assert result["end"]["state"]["env_step"] == 0
    schema = client.requests[0]["tools"][0]["parameters"]["properties"]["assignments"]
    assert schema["minItems"] == 2 and schema["maxItems"] == 4


@pytest.mark.parametrize("count", [2, 4])
def test_v4_native_rosters_solve_the_original_puzzle_and_replay(tmp_path, count):
    with open("config/eval_exploration_coordinated_v4.yaml") as stream:
        settings = yaml.safe_load(stream)
    case, = load_cases(settings, "unused")
    room, metadata = case

    def control(request):
        tool = request["tools"][0]["name"]
        view = json.loads(request["input_items"][-1]["content"])
        if tool == "spawn_subagents":
            assert view["min_subagents"] == 2 and view["max_subagents"] == 4
            assert view["roster_fixed_after_spawn"] is True
            return native(tool, dict(assignments=["Follow the offline test reference."] * count, rationale="Test roster."))
        assert "rooms" in view["task_board"] and "passages" in view["task_board"]
        assert all(code not in json.dumps(view) for code in room.codes.values())
        return native(tool, dict(assignments=[], rationale="Continue.", memory_updates=empty_updates(navigation=True)))

    controller = Orchestrator(Client(control), context_version="v4", history_turns=1)

    def factory(agent, env, assignment):
        actions = reference_for_team(metadata, env.agent_ids)["reference"]
        def respond(request):
            action = parse_action(actions[env.env_step][agent])
            schema = next(t for t in request["tools"] if t["name"] == action.verb)
            return native(action.verb, dict(zip(schema["parameters"]["required"], action.args)), f"{agent}-{env.env_step}")
        return LLMAgentPolicy(agent, Client(respond), navigation=True, shared_room=True,
                              coordination=True, teammates=env.agent_ids, assignment=assignment,
                              orchestration_protocol="v4", progress_memory=True, history_turns=3,
                              worker_broadcast=False)

    path = tmp_path / "native.jsonl"
    result = run_orchestrated_episode(room, controller, factory, path=path, metadata=metadata,
                                      turn_budget=60, worker_broadcast=False)
    assert result["end"]["reason"] == "success"
    assert result["metrics"]["joint_steps"] == 20
    assert result["events"][0]["metadata"]["num_subagents"] == count
    assert replay(path) == result["metrics"]


def test_v4_controller_relays_private_discovery_and_retains_report_backed_routes(tmp_path):
    case = generate_connected_episode(42)
    code = case.room.codes["box_code_1"]
    workers = {}
    def control(request):
        view = json.loads(request["input_items"][-1]["content"])
        name = request["tools"][0]["name"]
        if name == "spawn_subagents":
            assert code not in json.dumps(view)
            return native(name, dict(assignments=["Read clue_1 and report.", "Await relevant discoveries."], rationale="Two workers."))
        updates = empty_updates(navigation=True)
        tasks = []
        if view["step"] < 3:
            assert code not in json.dumps(view)
        if view["step"] == 3:
            report, = view["new_reports"]
            assert code in report["message"]
            reported_code = report["message"].split("code ")[1].split(";")[0]
            evidence = [report["evidence_id"]]
            updates["rooms"] = [dict(room_id="room_1", status="visited", evidence_ids=evidence),
                                dict(room_id="room_2", status="known", evidence_ids=evidence)]
            updates["passages"] = [dict(passage_id="passage_1", from_room="room_1", to_room="room_2",
                                        status="locked", evidence_ids=evidence)]
            tasks = [dict(agent_id="worker_2", task=f"Use reported code {reported_code} on box_1.")]
        if view["step"] == 4:
            assert view["task_board"]["rooms"][1]["room_id"] == "room_2"
            assert view["task_board"]["passages"][0]["to_room"] == "room_2"
        return native(name, dict(assignments=tasks, rationale="Relay reported discovery.", memory_updates=updates))

    controller = Orchestrator(Client(control), context_version="v4", history_turns=1)
    def factory(agent, env, assignment):
        def respond(request):
            if agent == "worker_1":
                action = parse_action([
                    "inspect(clue_1)",
                    f"report(In room_1: box_1 code {code}; passage_1 locked to room_2.)",
                    "wait()", "wait()",
                ][env.env_step])
            elif env.env_step == 2:
                reported_code = request["instructions"].split("Use reported code ")[1].split(" ")[0]
                action = parse_action(f"enter({reported_code}, box_1)")
            else:
                if env.env_step < 2:
                    assert code not in json.dumps(request["input_items"])
                action = parse_action("wait()")
            schema = next(t for t in request["tools"] if t["name"] == action.verb)
            return native(action.verb, dict(zip(schema["parameters"]["required"], action.args)), f"{agent}-{env.env_step}")
        client = Client(respond)
        workers[agent] = client
        return LLMAgentPolicy(agent, client, navigation=True, coordination=True, teammates=env.agent_ids,
                              assignment=assignment, orchestration_protocol="v4", worker_broadcast=False)

    path = tmp_path / "relay.jsonl"
    result = run_orchestrated_episode(case.room, controller, factory, path=path, turn_budget=10, max_steps=4,
                                      worker_broadcast=False)
    assert "box_1" in result["end"]["state"]["opened_nodes"]
    assert result["metrics"]["per_agent"]["worker_1"]["reports_sent"] == 1
    assert result["end"]["state"]["agents"]["worker_2"]["turns_left"] == 6
    assert len(controller.events) == 5  # One spawn and one coordination per tick.
    assert "ACTIVE WORKER GUIDANCE" in workers["worker_2"].requests[2]["instructions"]
    assert replay(path) == result["metrics"]


def test_navigation_memory_requires_provenance_persists_and_rejects_bad_delta_atomically():
    memory = CoordinationMemory(navigation=True)
    updates = empty_updates(navigation=True)
    updates["rooms"] = [dict(room_id="room_2", status="known", evidence_ids=["report_1"])]
    with pytest.raises(ActionParseError, match="evidence"):
        memory.apply(updates, ("worker_1", "worker_2"), step=1)
    memory.add_evidence("report_1", agent_id="worker_1", message="Open passage to room_2.", channel="report", step=1)
    memory.apply(updates, ("worker_1", "worker_2"), step=2)
    before = memory.view()
    assert before["rooms"][0]["verified"] is False
    memory.apply(empty_updates(navigation=True), ("worker_1", "worker_2"), step=3)
    assert memory.view() == before
    updates["rooms"][0]["room_id"] = None
    with pytest.raises(ActionParseError, match="identifier"):
        memory.apply(updates, ("worker_1", "worker_2"), step=4)
    assert memory.view() == before
    assert "rooms" not in memory_update_schema(("worker_1", "worker_2"))["properties"]


@pytest.mark.parametrize("kwargs", [dict(min_subagents=0), dict(min_subagents=True),
                                   dict(min_subagents=5, max_subagents=4)])
def test_bad_team_ranges_fail_early(kwargs):
    with pytest.raises(ValueError):
        OrchestratorLimits(**kwargs)


def test_new_defaults_and_config_validate_bounds_and_expose_assignments(tmp_path, capsys):
    assert (parse_args([]).min_subagents, parse_args([]).max_subagents) == (2, 4)
    assert parse_args([]).orchestrator_context_version == "v5"
    with pytest.raises(SystemExit):
        parse_args(["--min-subagents", "5", "--max-subagents", "4"])
    with open("config/eval_exploration_coordinated_v4.yaml") as stream:
        settings = yaml.safe_load(stream)
    cases = load_cases(settings, "unused")
    records = execute_plan(settings, make_plan(settings, cases, offline=True), tmp_path / "batch", offline=True)
    assert records[0]["metrics"]["joint_escape"]
    assert records[0]["metadata"]["min_subagents"] == 2
    assert records[0]["metadata"]["max_subagents"] == 4
    assert records[0]["metadata"]["orchestration_protocol"] == "v4"
    assert "Initial assignment" in capsys.readouterr().out
    rows = json.loads((tmp_path / "batch/summary.json").read_text())
    assert rows[0]["orchestration_protocol"] == "v4"
