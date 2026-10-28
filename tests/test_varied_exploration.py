"""Generated dependencies must stay solvable as placements and code roles vary."""
import json
import random

import pytest
import yaml

from motiverooms.connected_generator import ConnectedRoomsConfig
from motiverooms.env import MotiveRoomsEnv
from motiverooms.experiment import replay
from motiverooms.exploration_generator import generate_exploration_episode
from motiverooms.schema import Room
from scripts.evaluate import execute_plan, load_cases, make_plan
from scripts.rollout_llm import load_episode, parse_args


@pytest.mark.parametrize("count", [1, 2, 3, 4, 8, 20])
def test_varied_maps_have_executable_solutions_across_sizes_and_seeds(count):
    for seed in range(12):
        case = generate_exploration_episode(seed, ConnectedRoomsConfig(count))
        room = case.room
        assert case.metadata["generator_version"] == 2
        assert Room.from_dict(room.to_dict()) == room
        assert len(room.locations) == count
        assert all(p.requires is None for p in room.passages)
        assert room.node("final_exit").location == "room_1"
        assert len(room.codes["exit_code"]) == 2 * max(3, count)
        assert all(any(n.kind == "search" and n.location == location for n in room.nodes)
                   for location in room.locations)
        for parent in room.nodes:
            for child in parent.contents:
                if child not in room.items:
                    assert room.node(child).location == parent.location
        env = MotiveRoomsEnv(room, turn_budget=case.metadata["recommended_turn_budget"])
        assert room.codes["exit_code"] not in env.observe("A")
        for tick in case.metadata["reference"]:
            outcomes = env.step_joint(tick)
            assert all(o.valid and o.result != "wrong" for o in outcomes.values()), (count, seed, tick, outcomes)
        assert env.round_over and all(s.escaped for s in env.engine.agents.values())
        assert env.env_step == case.metadata["reference_joint_steps"]


def test_variation_changes_item_sources_code_targets_and_all_room_placements():
    before = random.getstate()
    cases = [generate_exploration_episode(seed, ConnectedRoomsConfig(4)) for seed in range(30)]
    assert random.getstate() == before
    assert generate_exploration_episode(42) == generate_exploration_episode(42)
    for item in ("desk_key", "brass_token"):
        assert len({c.metadata["item_sources"][item] for c in cases}) > 1
    assert {c.metadata["code_targets"]["shipping_receipt"] for c in cases} == {"supply_locker", "document_safe"}
    assert len({c.room.node("painting").location for c in cases}) == 4
    assert len({c.room.node("wall_plaque").text for c in cases}) > 1
    assert len({c.room.node("painting_inscription").text.split(":")[0] for c in cases}) > 1
    assert len({c.room.codes["exit_code"] for c in cases}) == len(cases)


def test_searches_hide_items_and_code_reading_stays_private():
    # One room isolates discovery/permissions from navigation.
    case = generate_exploration_episode(42, ConnectedRoomsConfig(1))
    room = case.room
    env = MotiveRoomsEnv(room, turn_budget=30)
    source = case.metadata["item_sources"]["desk_key"]
    assert not env.engine.step("A", "take(desk_key)").valid
    assert env.engine.step("A", f"inspect({source})").valid
    assert env.engine.step("A", "take(desk_key)").valid
    clue = room.node("shipping_receipt")
    parent = next(n for n in room.nodes if clue.id in n.contents)
    assert not env.engine.step("A", f"inspect({clue.id})").valid
    assert env.engine.step("A", f"inspect({parent.id})").valid
    assert env.engine.step("A", f"inspect({clue.id})").valid
    assert env.engine.agents["A"].read_codes[clue.id] == room.codes[clue.code]
    assert clue.id not in env.engine.agents["B"].read_codes
    target = case.metadata["code_targets"][clue.id]
    assert room.node(target).requires.code == clue.code
    assert target in clue.desc
    other = next(value for key, value in room.codes.items() if key not in {clue.code, "exit_code"})
    assert env.engine.step("A", f"enter({other}, {target})").result == "wrong"
    assert env.engine.step("A", f"enter({room.codes[clue.code]}, {target})").result != "wrong"


def test_default_rollout_uses_varied_exploration_and_preserves_explicit_sources():
    room, meta = load_episode(parse_args(["--num-rooms", "4", "--seed", "42"]))
    assert meta["generator_version"] == 2 and meta["generator"] == "exploration_rooms"
    assert len(room.locations) == 4 and room.node("painting")
    _, classic = load_episode(parse_args(["--seed", "42", "--exploration-version", "1"]))
    assert classic["generator_version"] == 1
    _, connected = load_episode(parse_args(["--seed", "42", "--room-source", "connected"]))
    assert connected["generator"] == "connected_rooms"
    with pytest.raises(SystemExit):
        parse_args(["--room", "rooms/examples/shared_mat.yaml", "--room-source", "exploration"])
    with pytest.raises(SystemExit):
        parse_args(["--room-source", "connected", "--exploration-version", "2"])


@pytest.mark.parametrize("version", [0, 3, True, "2"])
def test_unknown_generator_versions_are_rejected(version):
    with pytest.raises(ValueError, match="version"):
        generate_exploration_episode(42, version=version)


def test_varied_orchestrated_batch_replays_and_old_puzzles_remain_distinct(tmp_path):
    config = yaml.safe_load(open("config/eval_exploration_varied.yaml"))
    config.update(progress=False, num_rooms=4, seeds=[42])
    cases = load_cases(config, "unused")
    cases += load_cases(dict(config, exploration_version=1), "unused")
    plan = make_plan(config, cases, offline=True)
    records = execute_plan(config, plan, tmp_path / "batch", offline=True)
    assert len(records) == 2
    for record in records:
        assert record["metrics"]["joint_escape"] and record["metrics"]["api_calls"] == 0
        assert replay(record["path"]) == record["metrics"]
    summary = json.loads((tmp_path / "batch/summary.json").read_text())
    assert {row["profile"] for row in summary} == {"shared", "varied"}
