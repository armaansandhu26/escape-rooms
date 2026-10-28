"""Physical map validation, seeded generation, navigation and episode integration."""
from copy import deepcopy
import json
import random

import pytest

from motiverooms.actions import ActionParseError, parse_action
from motiverooms.benchmark_policies import ScriptedPolicy
from motiverooms.connected_generator import ConnectedRoomsConfig, generate_connected_episode
from motiverooms.env import MotiveRoomsEnv
from motiverooms.experiment import replay, run_episode, snapshot
from motiverooms.llm_policy import LLMAgentPolicy
from motiverooms.schema import Room, RoomValidationError, dump_room, load_room
from motiverooms.solver import solve_agent
from motiverooms.tool_calls import ToolCall, ToolResponse, tool_definitions
from scripts.evaluate import execute_plan, load_cases, make_plan
from scripts.rollout_llm import load_episode, parse_args


@pytest.fixture
def two_rooms():
    return Room.from_dict(dict(
        id="navigation_fixture", layout="shared", space="building",
        wings={"A": "building", "B": "building"},
        locations={"office": "Office", "storage": "Storage"},
        start_locations={"A": "office", "B": "office"},
        passages=[dict(id="hall_door", locations=["office", "storage"],
                       desc="Connecting door.", requires={"code": "hall_code"})],
        codes={"hall_code": "4721"}, items={"exit_key": {"name": "exit key"}},
        nodes=[
            dict(id="hall_card", wing="shared", kind="clue", code="hall_code", desc="Code for hall_door.", location="office"),
            dict(id="shelf", wing="shared", kind="spot", items=["exit_key"], desc="A shelf.", location="office"),
            dict(id="exit", wing="shared", kind="door", requires={"item": ["exit_key"]}, desc="Final exit.", location="storage"),
        ],
    ))


@pytest.mark.parametrize("count", [1, 2, 3, 7, 20])
def test_many_seeded_maps_have_valid_executable_solutions(count):
    for seed in range(12):
        case = generate_connected_episode(seed, ConnectedRoomsConfig(count))
        assert len(case.room.locations) == count
        assert len(case.room.passages) == count - 1
        assert sum(n.kind == "door" for n in case.room.nodes) == 1
        env = MotiveRoomsEnv(case.room, turn_budget=case.metadata["recommended_turn_budget"], communication="disabled")
        for tick in case.metadata["reference"]:
            outcomes = env.step_joint(tick)
            assert all(outcome.valid and outcome.result != "wrong" for outcome in outcomes.values())
        assert env.round_over
        assert all(state.escaped for state in env.engine.agents.values())
        assert env.env_step == case.metadata["reference_joint_steps"]
        assert {state.location for state in env.engine.agents.values()} <= set(case.room.locations)


def test_seed_controls_actual_placements_codes_and_lock_types_without_global_rng(tmp_path):
    before = random.getstate()
    first = generate_connected_episode(42)
    same = generate_connected_episode(42)
    assert random.getstate() == before
    assert first.room.to_dict() == same.room.to_dict()
    assert first.metadata == same.metadata
    cases = [generate_connected_episode(seed) for seed in range(10)]
    placements = {tuple(sorted((n.id, n.location, n.desc) for n in c.room.nodes)) for c in cases}
    codes = {tuple(sorted(c.room.codes.values())) for c in cases}
    locks = {tuple(p.requires.code is not None for p in c.room.passages) for c in cases}
    assert len(placements) > 1 and len(codes) == 10 and len(locks) > 1
    path = tmp_path / "episode.yaml"
    dump_room(first.room, path)
    assert load_room(path).to_dict() == first.room.to_dict()


@pytest.mark.parametrize("count", [0, -1, 2.5, True, "3"])
def test_bad_room_counts_fail_early(count):
    with pytest.raises(ValueError, match="positive integer"):
        ConnectedRoomsConfig(count)


@pytest.mark.parametrize("mutation", [
    lambda d: d["start_locations"].pop("B"),
    lambda d: d["start_locations"].update(A="missing"),
    lambda d: d["locations"].update(isolated="Isolated"),
    lambda d: d["nodes"][0].update(location="missing"),
    lambda d: d["nodes"][0].pop("location"),
    lambda d: d["passages"].append(dict(d["passages"][0], id="second_door")),
    lambda d: d["passages"][0].update(id="exit"),
    lambda d: d["passages"][0].update(locations=["office", "office"]),
    lambda d: d["passages"][0].update(requires={"code": "missing"}),
])
def test_invalid_maps_are_rejected(two_rooms, mutation):
    data = two_rooms.to_dict()
    mutation(data)
    with pytest.raises(RoomValidationError):
        Room.from_dict(data)


def test_locked_move_remote_objects_and_transfer_rules_cost_normal_turns(two_rooms):
    env = MotiveRoomsEnv(two_rooms, turn_budget=20)
    result = env.step_joint({"A": "move(storage)", "B": "take(exit_key)"})
    assert not result["A"].valid and "locked" in result["A"].result
    assert env.engine.agents["A"].location == "office"
    # Lock resolution precedes movement within the tick, preserving joint phases.
    result = env.step_joint({"A": "enter(4721, hall_door)", "B": "move(storage)"})
    assert all(o.valid for o in result.values())
    assert not env.round_over  # A connecting doorway is not the final exit.
    assert env.engine.agents["A"].location == "office"
    assert env.engine.agents["B"].location == "storage"
    result = env.step_joint({"A": "inspect(exit)", "B": "inspect(hall_card)"})
    assert all(not o.valid for o in result.values())
    result = env.step_joint({"A": "wait()", "B": "give(exit_key)"})
    assert not result["B"].valid and "same room" in result["B"].result
    assert "exit_key" in env.engine.agents["B"].bag
    assert "exit_key" not in env.engine.agents["A"].bag
    assert "same room" in env.engine.action_rejection_reason("B", parse_action("give(exit_key)"))
    result = env.step_joint({"A": "move(storage)", "B": "wait()"})
    assert result["A"].valid
    result = env.step_joint({"A": "wait()", "B": "give(exit_key)"})
    assert result["B"].valid and "exit_key" in env.engine.agents["A"].bag
    result = env.step_joint({"A": "use(exit_key, exit)", "B": "wait()"})
    assert env.round_over and all(s.escaped for s in env.engine.agents.values())
    assert all(s.turns_left == 13 for s in env.engine.agents.values())


def test_current_room_observations_keep_private_code_memory_and_bidirectional_passages(two_rooms):
    env = MotiveRoomsEnv(two_rooms, turn_budget=10)
    initial = env.observe("A")
    assert "CURRENT ROOM: office (Office)" in initial
    assert "room_id=storage" in initial
    assert "hall_card (type=clue)" in initial
    assert "exit (type=door)" not in initial
    assert "4721" not in initial
    env.step_joint({"A": "inspect(hall_card)", "B": "wait()"})
    env.step_joint({"A": "enter(4721, hall_door)", "B": "wait()"})
    env.step_joint({"A": "move(storage)", "B": "wait()"})
    observation = env.observe("A")
    assert "CURRENT ROOM: storage (Storage)" in observation
    assert "room_id=office" in observation
    assert "exit (type=door)" in observation
    assert "hall_card (type=clue)" not in observation
    assert "hall_card: 4721" in observation
    assert "4721" not in env.observe("B")
    result = env.step_joint({"A": "move(office)", "B": "wait()"})
    assert result["A"].valid
    result = env.step_joint({"A": "move(missing)", "B": "wait()"})
    assert not result["A"].valid


def test_solver_tracks_position_and_its_plan_executes(two_rooms):
    solution = solve_agent(two_rooms, "A").solo
    assert solution.turns == 5
    env = MotiveRoomsEnv(two_rooms, turn_budget=8)
    for action in solution.actions:
        result = env.engine.step("A", action)
        assert result.valid and result.result != "wrong"
    assert env.engine.agents["A"].escaped


def test_navigation_tools_are_explicit_and_legacy_rooms_keep_their_protocol():
    assert "move" not in {t["name"] for t in tool_definitions()}
    policy = LLMAgentPolicy("A", None, navigation=True)
    assert "move" in {t["name"] for t in policy.tools()}
    assert "CONNECTED ROOMS" in policy.system_prompt()
    assert str(ToolCall("one", "move", '{"room_id":"storage"}').action(navigation=True)) == "move(storage)"
    with pytest.raises(ActionParseError):
        ToolCall("one", "move", '{"room_id":"storage"}').action()
    env = MotiveRoomsEnv.from_yaml("rooms/examples/shared_mat.yaml")
    assert "locations" not in env.room.to_dict()
    assert all("location" not in state for state in snapshot(env)["agents"].values())


def test_native_tools_complete_generated_episode_and_replay(tmp_path):
    case = generate_connected_episode(42)
    env = MotiveRoomsEnv(case.room, turn_budget=case.metadata["recommended_turn_budget"])

    class Client:
        def __init__(self, agent):
            self.agent = agent
            self.requests = []
            self.index = 0

        def complete_tools(self, **kwargs):
            self.requests.append(deepcopy(kwargs))
            action = parse_action(case.metadata["reference"][self.index][self.agent])
            schema = next(t for t in kwargs["tools"] if t["name"] == action.verb)
            self.index += 1
            return ToolResponse((dict(type="function_call", name=action.verb,
                                      call_id=f"{self.agent}_{self.index}",
                                      arguments=json.dumps(dict(zip(schema["parameters"]["required"], action.args)))),))

    policies = {a: LLMAgentPolicy(a, Client(a), navigation=True, shared_room=True) for a in env.agent_ids}
    path = tmp_path / "native.jsonl"
    result = run_episode(env, policies, path=path, metadata=case.metadata, max_steps=env.turn_budget)
    assert result["end"]["reason"] == "success"
    assert replay(path) == result["metrics"]
    assert any("move(" in e.get("action", "") for e in result["events"] if e["type"] == "decision")
    assert result["events"][0]["metadata"]["seed"] == 42
    assert any(e["after"]["agents"]["A"]["location"] == "room_3"
               for e in result["events"] if e["type"] == "joint_step")


def test_generated_evaluation_variants_are_distinct_and_conditions_are_matched(tmp_path):
    config = dict(room_source="connected", num_rooms=3, episodes=2, seed=42, split="all",
                  conditions=["self_play"], both_seats=False, repeats=1,
                  communication_modes=["enabled", "disabled"], team_mode="fixed")
    cases = load_cases(config, "unused")
    assert [meta["seed"] for _, meta in cases] == [42, 43]
    assert cases[0][0].to_dict() != cases[1][0].to_dict()
    plan = make_plan(config, cases, offline=True)
    assert len(plan) == 4
    assert plan[0][0].to_dict() == plan[1][0].to_dict()
    records = execute_plan(config, plan, tmp_path / "evaluation", offline=True)
    assert len(records) == 4
    assert all(r["metrics"]["joint_escape"] for r in records)
    assert all(r["metrics"]["api_calls"] == 0 for r in records)
    assert all(replay(r["path"])["joint_escape"] for r in records)


def test_rollout_generates_fresh_seeds_and_allows_reproducible_size_overrides(monkeypatch):
    args = parse_args(["--num-rooms", "5", "--seed", "42"])
    first, metadata = load_episode(args)
    second, _ = load_episode(args)
    assert len(first.locations) == 5 and first.to_dict() == second.to_dict()
    assert metadata["seed"] == 42
    seeds = iter((100, 101))
    monkeypatch.setattr("scripts.rollout_llm.secrets.randbits", lambda _: next(seeds))
    assert load_episode(parse_args([]))[1]["seed"] == 100
    assert load_episode(parse_args([]))[1]["seed"] == 101
    legacy, _ = load_episode(parse_args(["--room", "rooms/examples/shared_mat.yaml"]))
    assert not legacy.locations
    with pytest.raises(SystemExit):
        parse_args(["--num-rooms", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--room", "rooms/examples/shared_mat.yaml", "--seed", "42"])


def test_reports_do_not_pool_different_map_sizes(tmp_path):
    config = dict(split="all", conditions=["self_play"], both_seats=False, repeats=1)
    cases = [generate_connected_episode(42, ConnectedRoomsConfig(size)) for size in (2, 3)]
    plan = make_plan(config, [(case.room, case.metadata) for case in cases], offline=True)
    execute_plan(config, plan, tmp_path / "sizes", offline=True)
    summaries = json.loads((tmp_path / "sizes/summary.json").read_text())
    assert len(summaries) == 2
    assert {row["num_rooms"] for row in summaries} == {2, 3}
    assert all(row["completed"] == 1 for row in summaries)
    assert "| Rooms |" in (tmp_path / "sizes/summary.md").read_text()
