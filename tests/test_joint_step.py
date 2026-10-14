from motiverooms.engine import RoomEngine
from motiverooms.schema import load_room

SHARED = "rooms/examples/shared_mat.yaml"


def test_joint_step_decrements_all_active_agents() -> None:
    engine = RoomEngine(load_room(SHARED), turn_budget=5)
    engine.step_joint(
        {
            "A": "inspect(sticky_note)",
            "B": "inspect(mat)",
        }
    )
    assert engine.agents["A"].turns_left == 4
    assert engine.agents["B"].turns_left == 4


def test_say_and_enter_same_tick() -> None:
    engine = RoomEngine(load_room(SHARED), turn_budget=8)
    engine.step_joint({"A": "inspect(wall_poster)", "B": "inspect(toolbox)"})
    outcomes = engine.step_joint(
        {
            "A": "say(The desk drawer code is 3906)",
            "B": "enter(3906, desk_drawer)",
        }
    )
    assert outcomes["A"].valid
    assert outcomes["B"].valid
    assert "desk_drawer" in engine.opened_nodes


def test_take_conflict_only_one_agent_gets_item() -> None:
    room = load_room(SHARED)
    engine = RoomEngine(room, turn_budget=8)
    engine.step_joint({"A": "inspect(wall_poster)", "B": "inspect(mat)"})
    engine.step_joint(
        {
            "A": "say(The desk drawer code is 3906)",
            "B": "enter(3906, desk_drawer)",
        }
    )
    engine.step_joint(
        {
            "A": "inspect(drawer_note)",
            "B": "inspect(filing_cabinet)",
        }
    )
    engine.step_joint(
        {
            "A": "say(The filing cabinet code is 4721)",
            "B": "enter(4721, filing_cabinet)",
        }
    )
    outcomes = engine.step_joint(
        {
            "A": "take(brass_key)",
            "B": "take(brass_key)",
        }
    )
    assert outcomes["A"].valid
    assert not outcomes["B"].valid
    assert outcomes["B"].result == "Someone else took it this step."
    assert "brass_key" in engine.agents["A"].bag
    assert "brass_key" not in engine.agents["B"].bag
