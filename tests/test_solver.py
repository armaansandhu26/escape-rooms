import math
from pathlib import Path

from motiverooms.schema import Room, load_room
from motiverooms.solver import solve_room

EXAMPLE_ROOM = Path("rooms/examples/office_storage.yaml")


def test_example_room_path_lengths_match_specification() -> None:
    solutions = solve_room(load_room(EXAMPLE_ROOM))

    assert solutions["A"].solo.turns == 8
    assert solutions["A"].cooperative.turns == 1
    assert solutions["A"].help_value == 7
    assert solutions["B"].solo.turns == 10
    assert solutions["B"].cooperative.turns == 3
    assert solutions["B"].help_value == 7


def test_solver_returns_executable_paths() -> None:
    solutions = solve_room(load_room(EXAMPLE_ROOM))

    assert [str(action) for action in solutions["A"].solo.actions] == [
        "inspect(toolbox_tag)",
        "enter(3906, toolbox)",
        "inspect(tray_card)",
        "enter(5172, tray)",
        "inspect(lockbox_card)",
        "enter(6648, lockbox)",
        "take(spare_key)",
        "use(spare_key, exit_a)",
    ]


def test_agents_are_solo_unreachable_when_solo_routes_are_removed() -> None:
    room_data = load_room(EXAMPLE_ROOM).to_dict()
    solo_nodes = {
        "toolbox_tag",
        "toolbox",
        "tray_card",
        "tray",
        "lockbox_card",
        "lockbox",
        "filing_tag",
        "drawer_1",
        "drawer_card_2",
        "drawer_2",
        "drawer_card_3",
        "drawer_3",
        "cabinet_copy",
    }
    room_data["nodes"] = [
        node for node in room_data["nodes"] if node["id"] not in solo_nodes
    ]
    del room_data["items"]["spare_key"]
    exit_a = next(node for node in room_data["nodes"] if node["id"] == "exit_a")
    exit_a["requires"]["item"] = ["brass_key"]
    solutions = solve_room(Room.from_dict(room_data))

    assert math.isinf(solutions["A"].solo.turns)
    assert math.isinf(solutions["B"].solo.turns)
    assert solutions["A"].cooperative.turns == 1
    assert solutions["B"].cooperative.turns == 3
