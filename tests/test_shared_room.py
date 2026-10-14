from pathlib import Path

from motiverooms.engine import RoomEngine
from motiverooms.partners import JOINT_SCRIPTS
from motiverooms.schema import load_room
from motiverooms.web_game import WebGame

SHARED_ROOM = Path("rooms/examples/shared_mat.yaml")


def test_shared_room_loads() -> None:
    room = load_room(SHARED_ROOM)
    assert room.layout == "shared"
    assert room.space == "room"
    assert len(room.nodes) == 10


def test_both_agents_see_the_same_objects() -> None:
    engine = RoomEngine(load_room(SHARED_ROOM), turn_budget=14)
    ids_a = {node.id for node in engine.visible_nodes("A")}
    ids_b = {node.id for node in engine.visible_nodes("B")}
    expected = {
        "mat",
        "coat_rack",
        "desk",
        "wall_poster",
        "desk_drawer",
        "filing_cabinet",
        "toolbox",
        "sticky_note",
        "exit",
    }
    assert ids_a == ids_b == expected
    assert "drawer_note" not in ids_a


def test_joint_script_has_no_waits() -> None:
    script = JOINT_SCRIPTS["shared_mat_example"]
    for step in script:
        assert all(action != "wait()" for action in step.values())


def test_joint_script_escapes_via_web_game() -> None:
    game = WebGame(load_room(SHARED_ROOM), turn_budget=14)
    for step in JOINT_SCRIPTS["shared_mat_example"]:
        active = set(game.active_agents)
        assert active == set(step.keys())
        outcomes = game.submit_joint(step)
        assert all(outcome.valid for outcome in outcomes.values()), outcomes
    assert game.engine.agents["A"].escaped
    assert game.engine.agents["B"].escaped
