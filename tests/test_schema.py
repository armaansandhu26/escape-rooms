from pathlib import Path

import pytest

from motiverooms.schema import Room, RoomValidationError, dump_room, load_room

EXAMPLE_ROOM = Path("rooms/examples/office_storage.yaml")


def test_example_room_loads_and_validates() -> None:
    room = load_room(EXAMPLE_ROOM)

    assert room.id == "office_storage_example"
    assert room.wings == {"A": "office", "B": "storage"}
    assert len(room.nodes) == 19
    assert room.node("exit_a").requires is not None
    assert room.node("exit_a").requires.items == ("brass_key", "spare_key")


def test_room_dump_round_trips(tmp_path: Path) -> None:
    room = load_room(EXAMPLE_ROOM)
    dumped_path = tmp_path / "room.yaml"

    dump_room(room, dumped_path)

    assert load_room(dumped_path) == room


def test_duplicate_code_values_are_rejected() -> None:
    room_data = load_room(EXAMPLE_ROOM).to_dict()
    room_data["codes"]["exit_b"] = room_data["codes"]["cabinet"]

    with pytest.raises(RoomValidationError, match="Code values must be unique"):
        Room.from_dict(room_data)


def test_unknown_content_is_rejected() -> None:
    room_data = load_room(EXAMPLE_ROOM).to_dict()
    toolbox = next(node for node in room_data["nodes"] if node["id"] == "toolbox")
    toolbox["contents"].append("does_not_exist")

    with pytest.raises(RoomValidationError, match="contains unknown id"):
        Room.from_dict(room_data)
