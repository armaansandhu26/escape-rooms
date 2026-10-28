"""Adaptive spawning, report privacy, supervision, accounting and replay."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from motiverooms.actions import ActionParseError, parse_action
from motiverooms.azure_openai import AzureOpenAISettings, AzureResponsesClient
from motiverooms.benchmark_policies import ScriptedPolicy
from motiverooms.connected_generator import ConnectedRoomsConfig, generate_connected_episode
from motiverooms.env import MotiveRoomsEnv
from motiverooms.experiment import replay
from motiverooms.llm_policy import LLMAgentPolicy
from motiverooms.orchestration import (
    Orchestrator, OrchestratorLimits, ScriptedOrchestrator,
    reference_for_team, run_orchestrated_episode, team_room,
)
from motiverooms.schema import load_room
from motiverooms.telemetry import RequestBudget
from motiverooms.tool_calls import ToolCall, ToolResponse
from scripts.evaluate import execute_plan, make_plan
from scripts.rollout_llm import parse_args


def native(name, arguments, call_id="control-call"):
    return ToolResponse((dict(type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id),))


def azure_transport(handler, budget=None):
    client = object.__new__(AzureResponsesClient)
    client.settings = AzureOpenAISettings("never-log-this", "https://example.com/openai/v1/", "test", max_output_tokens=1024)
    client.calls, client.budget = [], budget
    requests = []

    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        output = handler(kwargs)
        return SimpleNamespace(output=list(output.output), output_text=output.text, status="completed",
                               usage=dict(input_tokens=20, output_tokens=10, total_tokens=30))

    client._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return client, requests


def controller_client(count, budget=None, coordinate=None, order=None):
    def handler(request):
        name = request["tools"][0]["name"]
        view = json.loads(request["input"][-1]["content"])
        if order is not None:
            order.append(name)
        if name == "spawn_subagents":
            number = count(view) if callable(count) else count
            return native(name, dict(assignments=[f"Task for worker {i + 1}; report discoveries." for i in range(number)],
                                     rationale="Chosen from size and parallel work opportunities."))
        updates = coordinate(view) if coordinate else []
        return native(name, dict(assignments=updates, rationale="Coordinate from reported progress."), f"coord-{view['step']}")
    return azure_transport(handler, budget)


@pytest.mark.parametrize("chosen_count", [1, 2, 4])
def test_model_count_controls_actual_worker_instances_not_a_room_count_heuristic(tmp_path, chosen_count):
    case = generate_connected_episode(42)
    order, created = [], []
    budget = RequestBudget(max_calls=100, max_tokens=1000000)
    client, requests = controller_client(chosen_count, budget, order=order)
    controller = Orchestrator(client, OrchestratorLimits(5, min_subagents=1))

    def factory(agent, env, assignment):
        created.append(agent)
        def act(request):
            order.append(agent)
            return native("wait", {}, agent)
        worker, _ = azure_transport(act, budget)
        return LLMAgentPolicy(agent, worker, coordination=True, teammates=env.agent_ids,
                              solo=len(env.agent_ids) == 1, navigation=True, assignment=assignment)

    path = tmp_path / "roster.jsonl"
    result = run_orchestrated_episode(case.room, controller, factory, path=path, turn_budget=3, max_steps=1)
    assert created == [f"worker_{i + 1}" for i in range(chosen_count)]
    assert list(result["end"]["state"]["agents"]) == created
    assert order[:2] == ["spawn_subagents", "coordinate"]
    assert len(requests) == 2
    view = json.loads(requests[0]["input"][-1]["content"])
    assert view["num_rooms"] == 3 and view["max_subagents"] == 5
    assert set(view) == {"num_rooms", "min_subagents", "max_subagents", "turn_budget", "reward_mode", "shared_escape"}
    assert result["events"][0]["metadata"]["num_subagents"] == chosen_count
    assert result["metrics"]["orchestrator_calls"] == 2
    assert result["metrics"]["worker_calls"] == chosen_count
    assert budget.calls == 2 + chosen_count
    assert replay(path) == result["metrics"]
    assert "never-log-this" not in json.dumps(result)


def test_supervisor_sees_explicit_reports_only_and_updates_assignments_next_tick(tmp_path):
    case = generate_connected_episode(42)
    code = case.room.codes["box_code_1"]
    budget = RequestBudget(max_calls=100, max_tokens=1000000)
    def coordinate(view):
        if view["new_reports"]:
            return [dict(agent_id="worker_2", task=f"Use the reported code {code} for box_1.")]
        return []
    client, control_requests = controller_client(2, budget, coordinate)
    workers = {}
    scripts = {
        "worker_1": iter(["inspect(clue_1)", f"report(Code for box_1 is {code})", "wait()"]),
        "worker_2": iter(["wait()"] * 3),
    }
    def factory(agent, env, assignment):
        def act(request):
            action = parse_action(next(scripts[agent]))
            schema = next(tool for tool in request["tools"] if tool["name"] == action.verb)
            return native(action.verb, dict(zip(schema["parameters"]["required"], action.args)), f"{agent}-{len(workers[agent][1])}")
        worker_client, requests = azure_transport(act, budget)
        workers[agent] = (worker_client, requests)
        return LLMAgentPolicy(agent, worker_client, coordination=True, teammates=env.agent_ids,
                              navigation=True, assignment=assignment)
    path = tmp_path / "reports.jsonl"
    result = run_orchestrated_episode(case.room, Orchestrator(client), factory, path=path,
                                      turn_budget=10, max_steps=3)
    views = [json.loads(request["input"][-1]["content"]) for request in control_requests]
    assert code not in json.dumps(views[:3])
    assert views[3]["new_reports"] == [dict(agent_id="worker_1", message=f"Code for box_1 is {code}")]
    assert all("location" not in worker and "bag" not in worker for view in views[1:] for worker in view["workers"])
    assert code not in json.dumps(workers["worker_2"][1][:2])
    assert code in workers["worker_2"][1][2]["instructions"]
    assert result["end"]["state"]["agents"]["worker_1"]["turns_left"] == 7
    assert result["metrics"]["per_agent"]["worker_1"]["reports_sent"] == 1
    assert result["metrics"]["orchestrator_calls"] == 4
    assert result["metrics"]["worker_calls"] == 6
    assert budget.calls == 10
    assert len(result["end"]["state"]["reports"]) == 1
    assert replay(path) == result["metrics"]


@pytest.mark.parametrize("arguments", [
    {"assignments": [], "rationale": "none"},
    {"assignments": ["x"] * 4, "rationale": "too many"},
    {"assignments": "two", "rationale": "wrong type"},
    {"assignments": [42], "rationale": "wrong type"},
    {"assignments": ["  "], "rationale": "empty"},
    {"assignments": ["x"], "rationale": "ok", "extra": True},
])
def test_bad_spawn_is_logged_and_never_creates_workers_or_steps(tmp_path, arguments):
    case = generate_connected_episode(42)
    client, _ = azure_transport(lambda _: native("spawn_subagents", arguments))
    path = tmp_path / "bad.jsonl"
    result = run_orchestrated_episode(case.room, Orchestrator(client, OrchestratorLimits(3)),
                                      lambda *_: pytest.fail("Must not create workers"), path=path, turn_budget=5)
    assert result["end"]["reason"] == "policy_error"
    assert result["end"]["state"]["env_step"] == 0
    assert result["metrics"]["api_calls"] == 1
    assert result["events"][0]["metadata"]["num_subagents"] == 0
    assert any(e["type"] == "orchestrator_decision" and e["error"] for e in result["events"])
    assert replay(path) == result["metrics"]


@pytest.mark.parametrize("error", [TimeoutError("timeout"), RuntimeError("provider error")])
def test_controller_transport_failure_preserves_usage_and_zero_transitions(error):
    def handler(_):
        raise error
    budget = RequestBudget()
    client, _ = azure_transport(handler, budget)
    result = run_orchestrated_episode(generate_connected_episode(42).room, Orchestrator(client),
                                      lambda *_: pytest.fail("No workers after failed spawning"), turn_budget=5)
    assert result["end"]["reason"] == "api_error"
    assert result["end"]["state"]["env_step"] == 0
    assert result["metrics"]["unknown_usage_calls"] == 1
    assert result["metrics"]["orchestrator_calls"] == 1
    assert budget.calls == 1 and budget.tokens > 0


def test_controller_and_workers_share_the_call_budget():
    budget = RequestBudget(max_calls=1, max_tokens=100000)
    client, _ = controller_client(2, budget)
    case = generate_connected_episode(42)
    def factory(agent, env, task):
        worker, _ = azure_transport(lambda _: pytest.fail("Budget must stop before worker request"), budget)
        return LLMAgentPolicy(agent, worker, coordination=True, teammates=env.agent_ids, assignment=task)
    result = run_orchestrated_episode(case.room, Orchestrator(client), factory, turn_budget=5)
    assert result["end"]["reason"] == "budget_exceeded"
    assert result["end"]["state"]["env_step"] == 0
    assert result["metrics"]["orchestrator_calls"] == 1
    assert result["metrics"]["worker_calls"] == 0


def test_unknown_coordination_target_stops_before_any_worker_action():
    client, _ = controller_client(2, coordinate=lambda _: [dict(agent_id="invented_worker", task="act")])
    def factory(agent, env, task):
        return SimpleNamespace(assignment=task, choose_action=lambda _: pytest.fail("No action on invalid supervision"))
    result = run_orchestrated_episode(generate_connected_episode(42).room, Orchestrator(client), factory,
                                      turn_budget=5)
    assert result["end"]["reason"] == "policy_error"
    assert result["end"]["state"]["env_step"] == 0
    assert result["metrics"]["orchestrator_calls"] == 2


def test_large_team_explicit_transfers_and_reports_have_unambiguous_recipients():
    case = generate_connected_episode(42)
    controller = ScriptedOrchestrator()
    plan = controller.spawn(case.room, turn_budget=20, mode="mixed", shared_escape="together")
    plan = replace(plan, agents=("worker_1", "worker_2", "worker_3"), assignments=("a", "b", "c"))
    room = team_room(case.room, plan)
    env = MotiveRoomsEnv(room, turn_budget=20, coordination=True)
    item = next(iter(room.items))
    env.engine.agents["worker_1"].bag.add(item)
    outcome = env.engine.step("worker_1", f"give({item})")
    assert not outcome.valid and "give_to" in outcome.result
    outcome = env.engine.step("worker_1", f"give_to({item}, worker_3)")
    assert outcome.valid
    assert item in env.engine.agents["worker_3"].bag
    assert item not in env.engine.agents["worker_2"].bag
    outcome = env.engine.step("worker_3", "report(found something)")
    assert outcome.valid and not env.engine.messages
    assert env.engine.reports[-1].sender == "worker_3"
    observation = env.observe("worker_2")
    assert "worker_1" in observation and "worker_3" in observation
    assert "found something" not in observation
    policy = LLMAgentPolicy("worker_1", None, teammates=plan.agents, coordination=True)
    tools = {t["name"] for t in policy.tools()}
    assert {"give_to", "report"} <= tools and "give" not in tools
    assert str(ToolCall("one", "give_to", json.dumps(dict(item_id=item, agent_id="worker_2"))).action(team_size=3)) == f"give_to({item}, worker_2)"
    solo = LLMAgentPolicy("worker_1", None, solo=True, coordination=True, teammates=("worker_1",))
    assert "report" in {tool["name"] for tool in solo.tools()}
    assert "say" not in {tool["name"] for tool in solo.tools()}


def test_control_plane_cannot_bypass_disabled_communication():
    case = generate_connected_episode(42)
    with pytest.raises(ValueError, match="enabled communication"):
        MotiveRoomsEnv(case.room, coordination=True, communication="disabled")
    with pytest.raises(ValueError, match="enabled communication"):
        make_plan(dict(room_source="connected", split="all", conditions=["self_play"],
                       communication_modes=["disabled"]), [(case.room, case.metadata)])
    env = MotiveRoomsEnv(case.room)
    assert not env.engine.step("A", "report(no controller)").valid


@pytest.mark.parametrize("rooms,workers", [(2, 1), (5, 3), (8, 4)])
def test_offline_orchestrator_completes_and_replays_generated_maps(tmp_path, rooms, workers):
    case = generate_connected_episode(42, ConnectedRoomsConfig(rooms))
    controller = ScriptedOrchestrator(OrchestratorLimits(6, min_subagents=1))
    def factory(agent, env, _):
        return ScriptedPolicy(agent, env, reference_for_team(case.metadata, env.agent_ids), "reference")
    path = tmp_path / "offline.jsonl"
    result = run_orchestrated_episode(case.room, controller, factory, path=path,
                                      turn_budget=case.metadata["recommended_turn_budget"])
    assert result["end"]["reason"] == "success"
    assert len(result["end"]["state"]["agents"]) == workers
    assert result["metrics"]["api_calls"] == 0
    assert result["events"][0]["metadata"]["orchestrator_kind"] == "scripted"
    assert replay(path) == result["metrics"]


def test_split_wing_layouts_retain_required_seats_and_small_caps_fail():
    room = load_room("rooms/examples/office_storage.yaml")
    plan = ScriptedOrchestrator().spawn(room, turn_budget=14, mode="mixed", shared_escape="together")
    assert plan.agents == ("A", "B")
    assert team_room(room, plan) == room
    with pytest.raises(ValueError, match="physical seats"):
        ScriptedOrchestrator(OrchestratorLimits(1, min_subagents=1)).spawn(room, turn_budget=14, mode="mixed", shared_escape="together")


def test_orchestrated_batch_reports_actual_team_sizes_and_controller_kind(tmp_path):
    cases = [generate_connected_episode(42, ConnectedRoomsConfig(n)) for n in (2, 5)]
    config = dict(room_source="connected", split="all", conditions=["self_play"], repeats=1, both_seats=False,
                  min_subagents=1)
    plan = make_plan(config, [(c.room, c.metadata) for c in cases], offline=True)
    records = execute_plan(config, plan, tmp_path / "batch", offline=True)
    assert [r["metadata"]["num_subagents"] for r in records] == [1, 3]
    rows = json.loads((tmp_path / "batch/summary.json").read_text())
    assert {row["num_subagents"] for row in rows} == {1, 3}
    assert {row["team_mode"] for row in rows} == {"orchestrated"}
    assert {row["orchestrator_kind"] for row in rows} == {"scripted"}
    assert all(row["successful_episodes"] == 1 for row in rows)


def test_default_is_orchestrated_and_preview_does_not_invent_a_model_choice():
    assert parse_args([]).team_mode == "orchestrated"
    assert parse_args(["--max-subagents", "12"]).max_subagents == 12
    with pytest.raises(SystemExit):
        parse_args(["--max-subagents", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--decision-mode", "assisted"])


def test_existing_output_is_rejected_before_any_controller_request(tmp_path):
    path = tmp_path / "existing.jsonl"
    path.write_text("keep this\n")
    client, requests = controller_client(2)
    with pytest.raises(FileExistsError):
        run_orchestrated_episode(generate_connected_episode(42).room, Orchestrator(client),
                                  lambda *_: pytest.fail("No workers"), path=path)
    assert requests == [] and path.read_text() == "keep this\n"


def test_cli_default_completes_with_native_mocked_orchestrator_and_workers(tmp_path, monkeypatch, capsys):
    from scripts import rollout_llm
    case = generate_connected_episode(42, ConnectedRoomsConfig(2))
    offsets = {}
    def client_factory(settings, budget=None):
        def handler(request):
            name = request["tools"][0]["name"]
            if name == "spawn_subagents":
                return native(name, dict(assignments=["Solve the room and report progress.", "Assist and avoid duplicate work."],
                                         rationale="Two workers for this two-room test."))
            if name == "coordinate":
                sections = request["tools"][0]["parameters"]["properties"]["memory_updates"]["properties"]
                return native(name, dict(assignments=[], rationale="Continue the assigned tasks.",
                                         memory_updates={section: [] for section in sections}))
            agent = request["instructions"].splitlines()[0].removeprefix("You are agent ").removesuffix(".")
            index = offsets.get(agent, 0)
            offsets[agent] = index + 1
            action = parse_action(case.metadata["reference"][index]["A"] if agent == "worker_1" else "wait()")
            schema = next(tool for tool in request["tools"] if tool["name"] == action.verb)
            return native(action.verb, dict(zip(schema["parameters"]["required"], action.args)), f"{agent}-{index}")
        client, _ = azure_transport(handler, budget)
        client.settings = settings
        return client

    path = tmp_path / "cli.jsonl"
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test")
    monkeypatch.setattr(rollout_llm, "AzureResponsesClient", client_factory)
    monkeypatch.setattr(rollout_llm, "parse_args", lambda: parse_args([
        "--room-source", "connected", "--num-rooms", "2", "--seed", "42", "--output", str(path),
    ]))
    assert rollout_llm.main() == 0
    output = capsys.readouterr().out
    assert "Orchestrator created 2 subagents" in output
    assert "Team mode: orchestrated" in output
    metrics = replay(path)
    steps = case.metadata["reference_joint_steps"]
    assert metrics["joint_escape"] and metrics["orchestrator_calls"] == 1 + (steps - 1) // 5
    assert metrics["worker_calls"] == steps * 2


def test_v2_controller_gets_mechanics_and_start_but_no_puzzle_secrets():
    case = generate_connected_episode(42)
    client, requests = controller_client(2)
    controller = Orchestrator(client, context_version="v2", observe_team_chat=True, history_turns=2)
    controller.spawn(case.room, turn_budget=26, mode="collaborative", shared_escape="together")
    request = requests[0]
    view = json.loads(request["input"][-1]["content"])
    assert view["starting_arrangement"]["starting_room_id"] == "room_1"
    assert view["starting_arrangement"]["all_workers_together"] is True
    assert "There are no press, pull" in request["instructions"]
    assert "Do not discard a claim" in request["instructions"]
    assert "room_2" not in json.dumps(view)
    assert all(code not in json.dumps(view) for code in case.room.codes.values())


def test_v2_chat_visibility_remembers_claims_without_leaking_private_inspections():
    case = generate_connected_episode(42)
    client, requests = controller_client(2)
    controller = Orchestrator(client, context_version="v2", observe_team_chat=True, history_turns=1)
    plan = controller.spawn(case.room, turn_budget=20, mode="collaborative", shared_escape="together")
    env = MotiveRoomsEnv(team_room(case.room, plan), coordination=True, turn_budget=20)
    policies = {agent: SimpleNamespace(assignment=task) for agent, task in plan.tasks.items()}
    code = case.room.codes["box_code_1"]
    env.step_joint({"worker_1": "inspect(clue_1)", "worker_2": "wait()"})
    controller.coordinate(env, policies)
    assert code not in json.dumps(requests[-1]["input"])
    env.step_joint({"worker_1": f"say(I read {code} for box_1)", "worker_2": "wait()"})
    controller.coordinate(env, policies)
    view = json.loads(requests[-1]["input"][-1]["content"])
    assert view["new_team_messages"] == [dict(agent_id="worker_1", message=f"I read {code} for box_1")]
    assert view["remembered_claims"][0]["verified"] is False
    for _ in range(4):
        env.step_joint({"worker_1": "wait()", "worker_2": "wait()"})
        controller.coordinate(env, policies)
    view = json.loads(requests[-1]["input"][-1]["content"])
    assert view["new_team_messages"] == []
    assert code in json.dumps(view["remembered_claims"])
    assert sum(item.get("role") == "user" for item in requests[-1]["input"]) == 2


def test_bounded_worker_memory_keeps_observed_clue_mapping_and_attributed_chat():
    case = generate_connected_episode(42)
    env = MotiveRoomsEnv(case.room, turn_budget=12)
    sequence = iter(["inspect(clue_1)"] + ["wait()"] * 5)
    def handler(request):
        action = parse_action(next(sequence))
        schema = next(tool for tool in request["tools"] if tool["name"] == action.verb)
        return native(action.verb, dict(zip(schema["parameters"]["required"], action.args)))
    client, requests = azure_transport(handler)
    policy = LLMAgentPolicy("A", client, history_turns=1, navigation=True)
    for tick in range(6):
        _, action = policy.choose_action(env.observe("A"))
        outcome = env.step_joint({"A": action, "B": "say(I think clue_2 opens box_3)" if tick == 0 else "wait()"})
        policy.record_outcome(outcome["A"])
    last = requests[-1]["input"]
    assert sum(item.get("role") == "user" for item in last) == 2
    content = last[-1]["content"]
    assert "inspect(clue_1)" in content
    assert "code for box_1" in content and case.room.codes["box_code_1"] in content
    assert "B: I think clue_2 opens box_3" in content
    assert "claims, not verified facts" in content
