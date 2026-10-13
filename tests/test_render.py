from pathlib import Path

from motiverooms.engine import RoomEngine
from motiverooms.render import render_observation, render_system_prompt
from motiverooms.schema import load_room


def test_observation_is_a_fresh_state_summary() -> None:
    engine = RoomEngine(
        load_room(Path("rooms/examples/office_storage.yaml")),
        turn_budget=12,
    )
    engine.step("A", "inspect(sticky_note)")
    engine.step("B", "say(Do you have the cabinet code?)")

    observation = render_observation(engine, "A", round_number=2, total_rounds=5)

    assert "ROUND 2/5 | You are agent A | Turns left: 11" in observation
    assert "sticky_note" in observation
    assert "[read: code 4721]" in observation
    assert "tray_card" not in observation
    assert "- B: Do you have the cabinet code?" in observation
    assert "Reply with exactly one action on the last line." in observation


def test_system_prompt_explains_incentive() -> None:
    prompt = render_system_prompt("competitive")

    assert "points minus theirs" in prompt
    assert "exactly one action" in prompt
