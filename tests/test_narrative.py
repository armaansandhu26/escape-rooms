from pathlib import Path

from motiverooms.actions import Action
from motiverooms.engine import ActionOutcome, RoomEngine
from motiverooms.narrative import narrate_turn
from motiverooms.schema import load_room

SHARED_ROOM = Path("rooms/examples/shared_mat.yaml")


def test_narrate_mat_inspect() -> None:
    room = load_room(SHARED_ROOM)
    engine = RoomEngine(room, turn_budget=4)
    outcome = engine.step("A", "inspect(mat)")
    tag, line = narrate_turn(room, "A", "inspect(mat)", outcome, engine=engine)
    assert tag == "explore"
    assert "mat" in line
    assert "nothing" in line


def test_narrate_poster_after_sticky_hint() -> None:
    room = load_room(SHARED_ROOM)
    engine = RoomEngine(room, turn_budget=6)
    engine.step("A", "inspect(sticky_note)")
    outcome = engine.step("A", "inspect(wall_poster)")
    tag, line = narrate_turn(
        room, "A", "inspect(wall_poster)", outcome, engine=engine
    )
    assert tag == "explore"
    assert "3906" in line
    assert "hint" in line


def test_sticky_after_poster_is_redundant() -> None:
    room = load_room(SHARED_ROOM)
    engine = RoomEngine(room, turn_budget=6)
    engine.step("A", "inspect(wall_poster)")
    outcome = engine.step("B", "inspect(sticky_note)")
    tag, line = narrate_turn(
        room, "B", "inspect(sticky_note)", outcome, engine=engine
    )
    assert tag == "explore"
    assert "already" in line


def test_narrate_wrong_code() -> None:
    room = load_room(SHARED_ROOM)
    outcome = ActionOutcome(
        action=Action("enter", ("0000", "desk_drawer")),
        valid=True,
        result="wrong",
    )
    tag, line = narrate_turn(room, "A", "enter(0000, desk_drawer)", outcome)
    assert tag == "unlock"
    assert "wrong code" in line


def test_episode_log_includes_narrative() -> None:
    from motiverooms.catalog import preset_by_key, room_by_key
    from motiverooms.session import AppSession

    preset = preset_by_key("ideal_cooperation")
    session = AppSession(room_entry=room_by_key(preset.room_key), turn_budget=16)
    session.configure(
        room_key=preset.room_key,
        mode=preset.mode,
        partner_a="honest",
        partner_b="honest",
    )
    session.submit_auto()
    entry = next(r for r in session.episode_log if r.agent == "A")
    assert entry.tag == "explore"
    assert entry.narrative
    assert "poster" in entry.narrative.lower() or "sticky" in entry.narrative.lower()
