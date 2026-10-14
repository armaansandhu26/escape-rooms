from pathlib import Path

from motiverooms.schema import load_room
from motiverooms.web import STATIC_FILES
from motiverooms.web_game import WebGame

EXAMPLE_ROOM = Path("rooms/examples/office_storage.yaml")
SOLO_ROOM = Path("rooms/examples/office_solo.yaml")


def make_game(turn_budget: int = 12) -> WebGame:
    return WebGame(load_room(EXAMPLE_ROOM), turn_budget=turn_budget)


def test_web_game_joint_step_decrements_both_agents() -> None:
    game = make_game()

    outcomes = game.submit_joint(
        {
            "A": "inspect(sticky_note)",
            "B": "inspect(filing_tag)",
        }
    )

    assert outcomes["A"].valid
    assert outcomes["B"].valid
    assert game.active_agents == ("A", "B")
    assert game.engine.agents["A"].turns_left == 11
    assert game.engine.agents["B"].turns_left == 11


def test_invalid_joint_action_still_spends_the_tick() -> None:
    game = make_game()

    outcomes = game.submit_joint(
        {
            "A": "not an action",
            "B": "inspect(filing_tag)",
        }
    )

    assert not outcomes["A"].valid
    assert outcomes["B"].valid
    assert game.engine.agents["A"].turns_left == 11
    assert game.engine.agents["B"].turns_left == 11


def test_state_payload_contains_guided_action_data() -> None:
    payload = make_game().state_payload()

    assert payload["active_agents"] == ["A", "B"]
    assert payload["active_agent"] == "A"
    assert payload["agents"]["A"]["wing"] == "office"
    sticky_note = next(
        node for node in payload["view"]["nodes"] if node["id"] == "sticky_note"
    )
    assert sticky_note["kind"] == "clue"
    assert sticky_note["status"] == "Unread clue"
    assert 0 < sticky_note["position"]["x"] < 100
    assert 0 < sticky_note["position"]["y"] < 100


def test_spatial_positions_are_unique_and_stable() -> None:
    game = make_game()
    first_payload = game.state_payload()
    first_positions = {
        node["id"]: node["position"] for node in first_payload["view"]["nodes"]
    }

    game.submit_joint(
        {"A": "inspect(sticky_note)", "B": "inspect(filing_tag)"}
    )
    game.submit_joint({"A": "say(test)", "B": "inspect(filing_tag)"})
    second_payload = game.state_payload()
    second_positions = {
        node["id"]: node["position"] for node in second_payload["view"]["nodes"]
    }

    assert len({(value["x"], value["y"]) for value in first_positions.values()}) == len(
        first_positions
    )
    assert second_positions == first_positions


def test_reset_restores_initial_state() -> None:
    game = make_game()
    game.submit_joint(
        {"A": "inspect(sticky_note)", "B": "inspect(filing_tag)"}
    )
    game.submit_joint({"A": "say(test)", "B": "take(brass_key)"})

    game.reset()

    assert game.active_agent == "A"
    assert game.engine.agents["A"].turns_left == 12
    assert game.engine.agents["B"].bag == set()
    assert game.engine.messages == []


def test_opened_container_offers_take_for_items_inside() -> None:
    game = WebGame(load_room(SOLO_ROOM), turn_budget=12)
    game.submit("inspect(toolbox_tag)")
    game.submit("enter(3906, toolbox)")
    game.submit("inspect(tray_card)")
    game.submit("enter(5172, tray)")
    game.submit("inspect(lockbox_card)")
    game.submit("enter(6648, lockbox)")

    lockbox = next(
        node
        for node in game.state_payload()["view"]["nodes"]
        if node["id"] == "lockbox"
    )
    assert lockbox["opened"]
    assert lockbox["available_items"] == [{"id": "spare_key", "name": "spare key"}]

    take = game.submit("take(spare_key)")
    escape = game.submit("use(spare_key, exit_a)")
    assert take.valid
    assert escape.valid
    assert game.engine.agents["A"].escaped


def test_single_agent_room_marks_solo_mode_and_stays_on_one_agent() -> None:
    game = WebGame(load_room(SOLO_ROOM), turn_budget=12)

    assert game.single_agent
    payload = game.state_payload()
    assert payload["room"]["single_agent"] is True
    assert set(payload["agents"]) == {"A"}

    game.submit("inspect(toolbox_tag)")
    assert game.active_agent == "A"
    assert game.engine.agents["A"].turns_left == 11

    give = game.submit("give(spare_key)")
    assert not give.valid


def test_static_ui_files_are_present() -> None:
    static_directory = Path("motiverooms/static")

    for filename, _ in STATIC_FILES.values():
        assert (static_directory / filename).is_file()
