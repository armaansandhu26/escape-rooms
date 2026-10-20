from pathlib import Path

from motiverooms.engine import RoomEngine
from motiverooms.render import render_observation
from motiverooms.schema import load_room


def test_partner_last_step_after_joint_step() -> None:
    room = load_room(Path("rooms/examples/shared_mat.yaml"))
    engine = RoomEngine(room, turn_budget=12)
    engine.step_joint(
        {
            "A": "inspect(wall_poster)",
            "B": "inspect(drawer_note)",
        }
    )
    obs_a = render_observation(engine, "A", partner_actions="public")
    assert "PARTNER LAST STEP" in obs_a
    assert "inspect(drawer_note)" in obs_a
    assert "PARTNER RECENT" in obs_a


def test_open_desk_drawer_lists_inner_clue_not_item_keyerror() -> None:
    room = load_room(Path("rooms/examples/shared_mat.yaml"))
    engine = RoomEngine(room, turn_budget=12)
    engine.step_joint({"A": "enter(3906, desk_drawer)", "B": "wait()"})
    obs = render_observation(engine, "A")
    assert "desk_drawer" in obs
    assert "drawer_note (inspect clue)" in obs
    assert "[open; contains:" in obs


def test_open_container_lists_takeable_items() -> None:
    room = load_room(Path("rooms/examples/shared_mat.yaml"))
    engine = RoomEngine(room, turn_budget=12)
    engine.step_joint({"A": "enter(4721, filing_cabinet)", "B": "wait()"})
    obs = render_observation(engine, "A")
    assert "filing_cabinet" in obs
    assert "[open; contains: brass key (brass_key)]" in obs
    assert "take(<item_id>)" in obs
