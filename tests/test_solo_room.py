from pathlib import Path

from motiverooms.schema import load_room
from motiverooms.solver import solve_agent

SOLO_ROOM = Path("rooms/examples/office_solo.yaml")


def test_office_solo_loads_with_one_wing() -> None:
    room = load_room(SOLO_ROOM)
    assert room.wings == {"A": "office"}
    assert len(room.nodes) == 7


def test_office_solo_escape_path_is_eight_turns() -> None:
    solution = solve_agent(load_room(SOLO_ROOM), "A")
    assert solution.solo.turns == 8
    assert solution.cooperative.turns == 8
