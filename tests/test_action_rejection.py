from pathlib import Path

from motiverooms.actions import Action
from motiverooms.engine import RoomEngine
from motiverooms.joint_step import TAKE_CONFLICT_MESSAGE
from motiverooms.schema import load_room


def test_rejects_take_on_clue_node_id() -> None:
    engine = RoomEngine(load_room(Path("rooms/examples/shared_mat.yaml")), turn_budget=8)
    reason = engine.action_rejection_reason("A", Action("take", ("drawer_note",)))
    assert reason is not None
    assert "inspect" in reason


def test_rejects_enter_on_open_container() -> None:
    room = load_room(Path("rooms/examples/shared_mat.yaml"))
    engine = RoomEngine(room, turn_budget=8)
    engine.step_joint({"A": "enter(3906, desk_drawer)", "B": "wait()"})
    reason = engine.action_rejection_reason(
        "B", Action("enter", ("3906", "desk_drawer"))
    )
    assert reason == "That node is already open."


def test_rejects_take_when_partner_already_committed() -> None:
    room = load_room(Path("rooms/examples/shared_mat.yaml"))
    engine = RoomEngine(room, turn_budget=8)
    engine.step_joint({"A": "enter(4721, filing_cabinet)", "B": "wait()"})
    committed = {"A": Action("take", ("brass_key",))}
    reason = engine.action_rejection_reason(
        "B",
        Action("take", ("brass_key",)),
        committed=committed,
    )
    assert reason == TAKE_CONFLICT_MESSAGE
