"""Playable rooms and simulation presets for the browser UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _find_examples_dir() -> Path:
    module_dir = Path(__file__).resolve().parent
    for root in (module_dir.parent, module_dir, Path.cwd()):
        path = root / "rooms" / "examples"
        if path.is_dir():
            return path
    raise FileNotFoundError("Could not locate rooms/examples")


ROOT = _find_examples_dir()

CONTROLLER_CHOICES = ("human", "honest", "silent")
MODE_CHOICES = ("collaborative", "mixed", "competitive")


@dataclass(frozen=True)
class RoomEntry:
    key: str
    label: str
    path: Path
    single_agent: bool


@dataclass(frozen=True)
class PresetEntry:
    key: str
    label: str
    room_key: str
    partner_a: str
    partner_b: str
    mode: str


ROOMS: tuple[RoomEntry, ...] = (
    RoomEntry(
        key="shared_mat",
        label="Shared room (2 agents)",
        path=ROOT / "shared_mat.yaml",
        single_agent=False,
    ),
    RoomEntry(
        key="office_storage",
        label="Office × Storage (2 agents, split)",
        path=ROOT / "office_storage.yaml",
        single_agent=False,
    ),
    RoomEntry(
        key="office_solo",
        label="Office solo (1 agent)",
        path=ROOT / "office_solo.yaml",
        single_agent=True,
    ),
)

PRESETS: tuple[PresetEntry, ...] = (
    PresetEntry(
        key="human_play",
        label="Human play — control both agents",
        room_key="office_storage",
        partner_a="human",
        partner_b="human",
        mode="collaborative",
    ),
    PresetEntry(
        key="ideal_cooperation",
        label="Shared room — cooperative replay",
        room_key="shared_mat",
        partner_a="honest",
        partner_b="honest",
        mode="collaborative",
    ),
    PresetEntry(
        key="shared_competitive",
        label="Shared room — competitive replay",
        room_key="shared_mat",
        partner_a="honest",
        partner_b="honest",
        mode="competitive",
    ),
    PresetEntry(
        key="solo_human",
        label="Solo escape — human",
        room_key="office_solo",
        partner_a="human",
        partner_b="human",
        mode="mixed",
    ),
)


def room_by_key(key: str) -> RoomEntry:
    for room in ROOMS:
        if room.key == key:
            return room
    raise KeyError(f"Unknown room: {key}")


def preset_by_key(key: str) -> PresetEntry:
    for preset in PRESETS:
        if preset.key == key:
            return preset
    raise KeyError(f"Unknown preset: {key}")


def catalog_payload() -> dict:
    return {
        "api_version": 2,
        "features": {"peek_auto_step": True, "playback_speeds": True},
        "rooms": [
            {
                "key": room.key,
                "label": room.label,
                "single_agent": room.single_agent,
            }
            for room in ROOMS
        ],
        "presets": [
            {
                "key": preset.key,
                "label": preset.label,
                "room_key": preset.room_key,
                "partner_a": preset.partner_a,
                "partner_b": preset.partner_b,
                "mode": preset.mode,
            }
            for preset in PRESETS
        ],
        "controllers": list(CONTROLLER_CHOICES),
        "modes": list(MODE_CHOICES),
    }
