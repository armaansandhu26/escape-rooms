"""Report-backed controller memory, independent of room state and puzzle rules.

The model interprets reports. This module validates structure and provenance,
not whether a report or its interpretation is true. Updates are atomic deltas;
omitted entries persist until explicitly revised with supporting evidence.
"""
from __future__ import annotations

from copy import deepcopy

from motiverooms.actions import ActionParseError, IDENTIFIER_RE


SECTIONS = {
    "items": ("item_id", {"item_id", "holder", "room_id", "status"}, {"available", "held", "used", "unknown"}),
    "dependencies": ("target_id", {"target_id", "needed_item_id", "status"}, {"unresolved", "resolved", "unknown"}),
    "searches": ("node_id", {"node_id", "room_id", "status"}, {"useful", "exhausted", "blocked", "unknown"}),
    "locations": ("agent_id", {"agent_id", "room_id"}, None),
}
NAVIGATION_SECTIONS = {
    "rooms": ("room_id", {"room_id", "status"}, {"known", "visited", "surveyed"}),
    "passages": ("passage_id", {"passage_id", "from_room", "to_room", "status"}, {"open", "locked", "unknown"}),
}
EXIT_SECTIONS = {
    "exits": ("node_id", {"node_id", "room_id", "status", "code"}, {"locked", "open", "unknown"}),
}
MAX_UPDATES = 8
MAX_ENTRIES = 64


def _sections(navigation: bool, exits: bool = False) -> dict:
    sections = SECTIONS | NAVIGATION_SECTIONS if navigation else SECTIONS
    return sections | EXIT_SECTIONS if exits else sections


def empty_updates(*, navigation: bool = False, exits: bool = False) -> dict:
    return {name: [] for name in _sections(navigation, exits)}


def memory_update_schema(agents: tuple[str, ...], *, navigation: bool = False, exits: bool = False) -> dict:
    identifier = dict(type="string", pattern="^[A-Za-z][A-Za-z0-9_]*$")
    nullable_id = dict(type=["string", "null"], pattern=identifier["pattern"])
    properties = {}
    for name, (key, fields, statuses) in _sections(navigation, exits).items():
        row = {}
        for field in sorted(fields):
            if field == "status":
                row[field] = dict(type="string", enum=sorted(statuses))
            elif field == "holder":
                row[field] = dict(type=["string", "null"], enum=[*agents, None],
                                  description="Physical carrier, not reporter. Non-null only for status=held; otherwise null.")
            elif field == "agent_id":
                row[field] = dict(type="string", enum=list(agents))
            elif field == "code":
                row[field] = dict(type=["string", "null"], pattern="^[0-9]+$",
                                  description="Reported or assembled exit code, preserving leading zeros; null if unknown.")
            elif field in {"room_id", "needed_item_id"} and field != key:
                row[field] = dict(nullable_id)
            else:
                row[field] = dict(identifier)
        row["evidence_ids"] = dict(type="array", minItems=1, maxItems=8, items=dict(type="string"))
        properties[name] = dict(type="array", maxItems=MAX_UPDATES,
                                items=dict(type="object", properties=row,
                                           required=list(row), additionalProperties=False))
    return dict(type="object", properties=properties, required=list(properties), additionalProperties=False)


class CoordinationMemory:
    def __init__(self, *, navigation: bool = False, exits: bool = False):
        self._sections = _sections(navigation, exits)
        self._evidence: dict[str, dict] = {}
        self._entries: dict[str, dict[str | tuple[str, str | None], dict]] = {name: {} for name in self._sections}

    def add_evidence(self, evidence_id: str, *, agent_id: str, message: str, channel: str, step: int) -> dict:
        entry = dict(evidence_id=evidence_id, agent_id=agent_id, message=message, channel=channel, received_step=step)
        if evidence_id in self._evidence and self._evidence[evidence_id] != entry:
            raise ValueError("An evidence ID cannot be reused for a different report")
        self._evidence[evidence_id] = entry
        return deepcopy(entry)

    def apply(self, updates: dict, agents: tuple[str, ...], *, step: int) -> None:
        if not isinstance(updates, dict) or set(updates) != set(self._sections):
            raise ActionParseError("memory_updates requires " + ", ".join(self._sections))
        candidate = deepcopy(self._entries)
        for section, (key, fields, statuses) in self._sections.items():
            rows = updates[section]
            if not isinstance(rows, list) or len(rows) > MAX_UPDATES:
                raise ActionParseError(f"Too many {section} memory updates")
            seen = set()
            for row in rows:
                if not isinstance(row, dict) or set(row) != fields | {"evidence_ids"}:
                    raise ActionParseError(f"Invalid {section} memory fields")
                for field in fields - {"status", "holder", "code"}:
                    value = row[field]
                    if value is None and field in {"room_id", "needed_item_id"} and field != key:
                        continue
                    if not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value):
                        raise ActionParseError(f"Invalid memory identifier: {field}")
                if statuses is not None and (not isinstance(row["status"], str) or row["status"] not in statuses):
                    raise ActionParseError(f"Invalid {section} memory status")
                if section == "items":
                    if row["holder"] is not None and row["holder"] not in agents:
                        raise ActionParseError("Memory holder must be a team member or null")
                    if (row["status"] == "held") != (row["holder"] is not None):
                        raise ActionParseError("Only held items name a holder; held items require one")
                if section == "locations" and row["agent_id"] not in agents:
                    raise ActionParseError("Memory location requires a team member")
                if section == "exits" and row["code"] is not None:
                    code = row["code"]
                    if not isinstance(code, str) or not code.isascii() or not code.isdigit():
                        raise ActionParseError("Exit code must be a numeric string or null")
                sources = row["evidence_ids"]
                if (not isinstance(sources, list) or not 1 <= len(sources) <= 8
                        or any(not isinstance(source, str) or source not in self._evidence for source in sources)):
                    raise ActionParseError("Memory entries must cite received evidence IDs")
                entry_key = row[key]
                # A mechanism can depend on more than one item.
                if section == "dependencies":
                    entry_key = (row[key], row["needed_item_id"])
                if entry_key in seen:
                    raise ActionParseError("Duplicate memory updates for the same entry")
                seen.add(entry_key)
                candidate[section][entry_key] = dict(deepcopy(row), updated_step=step, verified=False)
            if len(candidate[section]) > MAX_ENTRIES:
                raise ActionParseError(f"The {section} memory board is full")
        self._entries = candidate

    def view(self) -> dict:
        return {name: deepcopy(list(entries.values())) for name, entries in self._entries.items()}

    def supporting_evidence(self) -> list[dict]:
        ids = {source for entries in self._entries.values() for row in entries.values() for source in row["evidence_ids"]}
        ids.update(list(self._evidence)[-16:])
        return [deepcopy(evidence) for key, evidence in self._evidence.items() if key in ids]
