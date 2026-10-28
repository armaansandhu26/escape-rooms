"""Managers check in on reports or per-worker silence, while workers keep acting."""
from copy import deepcopy
import json

import pytest
import yaml

from motiverooms.actions import parse_action
from motiverooms.connected_generator import generate_connected_episode
from motiverooms.coordination_memory import empty_updates
from motiverooms.coordination_memory import CoordinationMemory
from motiverooms.actions import ActionParseError
from motiverooms.schema import Room
from motiverooms.env import MotiveRoomsEnv
from motiverooms.experiment import replay
from motiverooms.llm_policy import LLMAgentPolicy
from motiverooms.orchestration import ManagerSchedule, Orchestrator, run_orchestrated_episode, reference_for_team, team_room
from motiverooms.tool_calls import ToolResponse
from scripts.evaluate import execute_plan, load_cases, make_plan


def native(name, arguments, id="call"):
    return ToolResponse((dict(type="function_call", name=name, arguments=json.dumps(arguments), call_id=id),))


class Client:
    def __init__(self, count=2, callback=None):
        self.count, self.callback = count, callback
        self.requests = []

    def complete_tools(self, **request):
        self.requests.append(deepcopy(request))
        view = json.loads(request["input_items"][-1]["content"])
        name = request["tools"][0]["name"]
        assert "coordinate before each simultaneous worker tick" not in request["instructions"]
        assert "MANAGER SUPERVISION" in request["instructions"]
        if name == "spawn_subagents":
            return native(name, dict(assignments=["Explore your objective independently."] * self.count,
                                     rationale="Two to four autonomous workers."))
        updates = self.callback(view) if self.callback else []
        return native(name, dict(assignments=updates, rationale="Manager check-in.",
                                 memory_updates=empty_updates(navigation=True, exits=True)), f"c{view['step']}")


def test_reports_do_not_hide_other_workers_silence_and_checkins_do_not_repeat_each_tick():
    schedule = ManagerSchedule(5)
    active = ("worker_1", "worker_2")
    assert schedule.due(0, active, ()) is None
    report = schedule.due(3, active, ("worker_1",))
    assert report["reasons"] == ["new_reports"]
    schedule.record(3, ("worker_1",), [])
    assert schedule.due(4, active, ()) is None
    due = schedule.due(5, active, ())
    assert due["overdue_workers"] == ["worker_2"]
    schedule.record(5, (), due["overdue_workers"])
    assert schedule.due(6, active, ()) is None
    assert schedule.due(8, active, ())["overdue_workers"] == ["worker_1"]
    # A new report from a due worker resolves its silence trigger.
    assert schedule.due(8, active, ("worker_1",))["overdue_workers"] == []
    assert schedule.due(5, ("worker_1",), ()) is None  # inactive workers aren't checked


def test_workers_act_every_tick_while_manager_only_checks_every_five_silent_ticks(tmp_path):
    case = generate_connected_episode(42)
    client = Client()
    controller = Orchestrator(client, context_version="v5", history_turns=1, check_in_interval=5)
    created = {}
    def factory(agent, env, assignment):
        class Worker:
            def __init__(self):
                self.assignment = assignment
                self.calls = 0

            def choose_action(self, observation):
                self.calls += 1
                assert self.assignment == assignment  # No replacements between check-ins.
                return "wait()", "wait()"
        created[agent] = Worker()
        return created[agent]
    path = tmp_path / "silence.jsonl"
    result = run_orchestrated_episode(case.room, controller, factory, path=path, turn_budget=20, max_steps=12)
    assert [event["step"] for event in controller.events] == [0, 6, 11]
    assert [worker.calls for worker in created.values()] == [12, 12]
    assert all(state["turns_left"] == 8 for state in result["end"]["state"]["agents"].values())
    assert len(client.requests) == 3
    assert result["events"][0]["metadata"]["supervision_mode"] == "manager"
    assert replay(path) == result["metrics"]


def test_reports_trigger_next_tick_help_but_private_actions_do_not(tmp_path):
    case = generate_connected_episode(42)
    code = case.room.codes["box_code_1"]
    def callback(view):
        assert view["check_in"]["reasons"] == ["new_reports"]
        assert view["step"] == 3
        assert code in view["new_reports"][0]["message"]
        return [dict(agent_id="worker_2", task=f"Open box_1 with reported code {code}.")]
    client = Client(callback=callback)
    controller = Orchestrator(client, context_version="v5", check_in_interval=5)
    workers = {}
    def factory(agent, env, assignment):
        class Worker:
            def __init__(self):
                self.assignment = assignment

            def choose_action(self, observation):
                if agent == "worker_1":
                    action = ["inspect(clue_1)", f"report(Code for box_1 is {code}.)", "wait()", "wait()"][env.env_step]
                else:
                    if env.env_step < 2:
                        assert code not in self.assignment and code not in observation
                    action = f"enter({code}, box_1)" if env.env_step == 2 else "wait()"
                    if env.env_step == 2:
                        assert code in self.assignment
                return action, action
        workers[agent] = Worker()
        return workers[agent]
    path = tmp_path / "reports.jsonl"
    result = run_orchestrated_episode(case.room, controller, factory, path=path, turn_budget=10, max_steps=4)
    assert [event["step"] for event in controller.events] == [0, 3]
    assert code not in json.dumps(client.requests[0]["input_items"])
    assert "box_1" in result["end"]["state"]["opened_nodes"]
    assert replay(path) == result["metrics"]


def test_report_batches_and_observed_chat_trigger_a_single_checkin():
    case = generate_connected_episode(42)
    client = Client()
    manager = Orchestrator(client, context_version="v5", observe_team_chat=True)
    plan = manager.spawn(case.room, turn_budget=20, mode="collaborative", shared_escape="together")
    env = MotiveRoomsEnv(team_room(case.room, plan), coordination=True, turn_budget=20)
    policies = {agent: type("Policy", (), {"assignment": "Explore."})() for agent in plan.agents}
    env.step_joint({"worker_1": "report(Found a blocker.)", "worker_2": "report(Found an open route.)"})
    assert manager.coordinate(env, policies)
    assert len(client.requests) == 2
    assert len(manager.events[-1]["input"]["new_reports"]) == 2
    assert not manager.coordinate(env, policies)
    env.step_joint({"worker_1": "say(Useful public discovery.)", "worker_2": "wait()"})
    assert manager.coordinate(env, policies)
    assert manager.events[-1]["input"]["check_in"]["reasons"] == ["team_chat"]


@pytest.mark.parametrize("interval", [0, -1, True, "5"])
def test_invalid_checkin_intervals_are_rejected(interval):
    with pytest.raises(ValueError, match="positive integer"):
        ManagerSchedule(interval)
    with pytest.raises(ValueError, match="positive integer"):
        make_plan(dict(orchestrator_check_in_interval=interval), [])


@pytest.mark.parametrize("count", [2, 4])
def test_native_workers_complete_puzzle_with_sparse_manager_calls_and_replay(tmp_path, count):
    with open("config/eval_exploration_manager_v5.yaml") as stream:
        config = yaml.safe_load(stream)
    room, metadata = load_cases(config, "unused")[0]
    client = Client(count)
    controller = Orchestrator(client, context_version="v5", history_turns=1)
    def factory(agent, env, assignment):
        reference = reference_for_team(metadata, env.agent_ids)["reference"]
        class WorkerClient:
            def complete_tools(self, **request):
                action = parse_action(reference[env.env_step][agent])
                schema = next(t for t in request["tools"] if t["name"] == action.verb)
                assert "AUTONOMOUS WORK BETWEEN MANAGER CHECK-INS" in request["instructions"]
                return native(action.verb, dict(zip(schema["parameters"]["required"], action.args)), f"{agent}-{env.env_step}")
        return LLMAgentPolicy(agent, WorkerClient(), coordination=True, teammates=env.agent_ids,
                              assignment=assignment, navigation=True, shared_room=True,
                              orchestration_protocol="v5", history_turns=3, progress_memory=True)
    path = tmp_path / "native.jsonl"
    result = run_orchestrated_episode(room, controller, factory, path=path, metadata=metadata, turn_budget=60)
    assert result["end"]["reason"] == "success"
    assert len(client.requests) == 4  # Spawn plus check-ins at ticks 6, 11 and 16.
    assert result["metrics"]["joint_steps"] == 20
    assert replay(path) == result["metrics"]


def test_manager_config_executes_and_reports_its_schedule(tmp_path, capsys):
    with open("config/eval_exploration_manager_v5.yaml") as stream:
        config = yaml.safe_load(stream)
    cases = load_cases(config, "unused")
    records = execute_plan(config, make_plan(config, cases, offline=True), tmp_path / "manager", offline=True)
    output = capsys.readouterr().out
    assert output.count("Manager check-in") == 3
    assert "tick 6: silent_workers" in output
    assert records[0]["metadata"]["check_in_interval"] == 5
    summary = json.loads((tmp_path / "manager/summary.json").read_text())
    assert summary[0]["supervision_mode"] == "manager" and summary[0]["check_in_interval"] == 5


def test_exit_reports_are_briefed_to_every_worker_and_route_code_holder_to_exit(tmp_path):
    room = Room.from_dict(dict(
        id="exit_relay", layout="shared", space="building", wings={"A": "building", "B": "building"},
        locations={"room_1": "Office", "room_2": "Exit hall"},
        start_locations={"A": "room_1", "B": "room_1"},
        codes={"exit_code": "0042"}, items={},
        passages=[dict(id="hall", locations=["room_1", "room_2"], desc="An open passage.")],
        nodes=[dict(id="exit_card", wing="shared", kind="clue", desc="The final exit code.", code="exit_code", location="room_1"),
               dict(id="final_exit", wing="shared", kind="door", desc="The building exit.",
                    requires={"code": "exit_code"}, location="room_2")],
    ))
    class ExitManager:
        def __init__(self):
            self.requests = []

        def complete_tools(self, **request):
            self.requests.append(deepcopy(request))
            name = request["tools"][0]["name"]
            view = json.loads(request["input_items"][-1]["content"])
            if name == "spawn_subagents":
                assert "0042" not in json.dumps(view) and "final_exit" not in json.dumps(view)
                return native(name, dict(assignments=["Find and report the code.", "Explore and report the exit location."], rationale="Split discoveries."))
            assert view["step"] == 3
            reports = {report["agent_id"]: report for report in view["new_reports"]}
            code = reports["worker_1"]["message"].split()[-1]
            _, node, _, location = reports["worker_2"]["message"].split()
            updates = empty_updates(navigation=True, exits=True)
            updates["exits"] = [dict(node_id=node, room_id=location, code=code, status="locked",
                                     evidence_ids=[report["evidence_id"] for report in reports.values()])]
            return native(name, dict(assignments=[dict(agent_id="worker_1", task="Finish using the persistent exit briefing.")],
                                     rationale="Relay code and destination to the team.", memory_updates=updates))

    manager_client = ExitManager()
    manager = Orchestrator(manager_client, context_version="v5")
    policies, clients = {}, {}
    def factory(agent, env, assignment):
        class WorkerClient:
            def __init__(self):
                self.requests = []

            def complete_tools(self, **request):
                self.requests.append(deepcopy(request))
                if env.env_step < 2:
                    if agent == "worker_1":
                        line = ["inspect(exit_card)", "report(Exit code 0042)"][env.env_step]
                    else:
                        assert "0042" not in json.dumps(request)
                        line = ["move(room_2)", "report(Exit final_exit in room_2)"][env.env_step]
                else:
                    briefing = json.loads(request["instructions"].split(
                        "MANAGER EXIT BRIEFING (reported, unverified)\n")[1].splitlines()[0])[0]
                    assert briefing["room_id"] == "room_2" and briefing["code"] == "0042"
                    if agent == "worker_1":
                        line = f"move({briefing['room_id']})" if env.env_step == 2 else f"enter({briefing['code']}, {briefing['node_id']})"
                    else:
                        line = "wait()"
                action = parse_action(line)
                schema = next(t for t in request["tools"] if t["name"] == action.verb)
                return native(action.verb, dict(zip(schema["parameters"]["required"], action.args)), f"{agent}-{env.env_step}")
        clients[agent] = WorkerClient()
        policies[agent] = LLMAgentPolicy(agent, clients[agent], coordination=True, teammates=env.agent_ids,
                                       assignment=assignment, navigation=True, shared_room=True, orchestration_protocol="v5")
        return policies[agent]
    path = tmp_path / "exit.jsonl"
    result = run_orchestrated_episode(room, manager, factory, path=path, turn_budget=12, worker_broadcast=False)
    assert result["end"]["reason"] == "success" and result["metrics"]["joint_steps"] == 4
    assert len(manager_client.requests) == 2  # Only spawn and report-triggered help.
    assert policies["worker_1"].exit_guidance == policies["worker_2"].exit_guidance
    assert policies["worker_1"].exit_guidance is not policies["worker_2"].exit_guidance
    assert result["end"]["state"]["agents"]["worker_2"]["read_codes"] == {}
    assert replay(path) == result["metrics"]


def test_exit_briefing_persists_location_before_code_and_rejects_invalid_deltas():
    memory = CoordinationMemory(navigation=True, exits=True)
    updates = empty_updates(navigation=True, exits=True)
    updates["exits"] = [dict(node_id="final_exit", room_id="room_1", code=None, status="locked", evidence_ids=["report_1"])]
    with pytest.raises(ActionParseError, match="evidence"):
        memory.apply(updates, ("worker_1", "worker_2"), step=1)
    memory.add_evidence("report_1", agent_id="worker_1", message="Exit final_exit in room_1.", channel="report", step=1)
    memory.apply(updates, ("worker_1", "worker_2"), step=2)
    assert memory.view()["exits"][0]["code"] is None
    memory.apply(empty_updates(navigation=True, exits=True), ("worker_1", "worker_2"), step=3)
    before = memory.view()
    updates["exits"][0]["code"] = 42
    with pytest.raises(ActionParseError, match="numeric string"):
        memory.apply(updates, ("worker_1", "worker_2"), step=4)
    assert memory.view() == before
