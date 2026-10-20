from pathlib import Path

from motiverooms.env import MotiveRoomsEnv
from motiverooms.engine import RoomEngine
from motiverooms.schema import load_room

SHARED = Path("rooms/examples/shared_mat.yaml")


def test_competitive_shared_room_race_escape() -> None:
    env = MotiveRoomsEnv.from_yaml(
        SHARED, turn_budget=12, mode="competitive", shared_escape="race",
    )
    assert env.engine.shared_escape == "race"
    env.engine.agents["A"].bag.add("brass_key")
    env.step_joint({"A": "use(brass_key, exit)", "B": "inspect(mat)"})
    assert env.engine.agents["A"].escaped
    assert not env.engine.agents["B"].escaped
    assert env.round_over


def test_collaborative_shared_room_still_escapes_together() -> None:
    engine = RoomEngine(load_room(SHARED), turn_budget=12, shared_escape="together")
    engine.agents["A"].bag.add("brass_key")
    engine.step_joint({"A": "use(brass_key, exit)", "B": "wait()"})
    assert engine.agents["A"].escaped
    assert engine.agents["B"].escaped


def test_shared_race_option_does_not_end_split_room_after_one_escape() -> None:
    env = MotiveRoomsEnv.from_yaml(
        "rooms/examples/office_storage.yaml", shared_escape="race",
    )
    env.engine.agents["A"].bag.add("brass_key")
    env.step_joint({"A": "use(brass_key, exit_a)", "B": "wait()"})
    assert not env.round_over
    assert env.active_agents() == ("B",)
