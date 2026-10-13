"""Room data structures and strict YAML validation.

The schema deliberately uses plain dataclasses.  They make room state easy to
inspect in a debugger and keep the environment independent of validation
frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

AgentId = Literal["A", "B"]
NodeKind = Literal["clue", "spot", "container", "door"]
NODE_KINDS = {"clue", "spot", "container", "door"}


class RoomValidationError(ValueError):
    """Raised when a room cannot be represented or safely played."""


@dataclass(frozen=True)
class Item:
    id: str
    name: str


@dataclass(frozen=True)
class Requirement:
    """Exactly one code or one-or-more interchangeable items."""

    code: str | None = None
    items: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Requirement":
        unknown = set(data) - {"code", "item"}
        if unknown:
            raise RoomValidationError(
                f"Unknown requirement field(s): {', '.join(sorted(unknown))}"
            )
        raw_items = data.get("item", [])
        if isinstance(raw_items, str):
            raw_items = [raw_items]
        if not isinstance(raw_items, list):
            raise RoomValidationError("Requirement 'item' must be a list")
        return cls(code=data.get("code"), items=tuple(raw_items))

    def to_dict(self) -> dict[str, Any]:
        if self.code is not None:
            return {"code": self.code}
        return {"item": list(self.items)}


@dataclass(frozen=True)
class Node:
    id: str
    wing: AgentId
    kind: NodeKind
    desc: str
    hidden: bool = False
    code: str | None = None
    items: tuple[str, ...] = ()
    requires: Requirement | None = None
    contents: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Node":
        required = {"id", "wing", "kind", "desc"}
        missing = required - set(data)
        if missing:
            raise RoomValidationError(
                f"Node is missing field(s): {', '.join(sorted(missing))}"
            )
        allowed = required | {"hidden", "code", "items", "requires", "contents"}
        unknown = set(data) - allowed
        if unknown:
            raise RoomValidationError(
                f"Node {data['id']!r} has unknown field(s): "
                f"{', '.join(sorted(unknown))}"
            )

        requirement = data.get("requires")
        return cls(
            id=str(data["id"]),
            wing=data["wing"],
            kind=data["kind"],
            desc=str(data["desc"]),
            hidden=bool(data.get("hidden", False)),
            code=data.get("code"),
            items=tuple(data.get("items", ())),
            requires=(
                Requirement.from_dict(requirement)
                if isinstance(requirement, Mapping)
                else None
            ),
            contents=tuple(data.get("contents", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "wing": self.wing,
            "kind": self.kind,
            "desc": self.desc,
        }
        if self.hidden:
            data["hidden"] = True
        if self.code is not None:
            data["code"] = self.code
        if self.items:
            data["items"] = list(self.items)
        if self.requires is not None:
            data["requires"] = self.requires.to_dict()
        if self.contents:
            data["contents"] = list(self.contents)
        return data


@dataclass(frozen=True)
class Room:
    id: str
    wings: dict[AgentId, str]
    codes: dict[str, str]
    items: dict[str, Item]
    nodes: tuple[Node, ...]
    _nodes_by_id: dict[str, Node] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_nodes_by_id", {node.id: node for node in self.nodes})
        self.validate()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Room":
        required = {"id", "wings", "codes", "items", "nodes"}
        missing = required - set(data)
        if missing:
            raise RoomValidationError(
                f"Room is missing field(s): {', '.join(sorted(missing))}"
            )
        unknown = set(data) - required
        if unknown:
            raise RoomValidationError(
                f"Room has unknown field(s): {', '.join(sorted(unknown))}"
            )

        raw_items = data["items"]
        if not isinstance(raw_items, Mapping):
            raise RoomValidationError("'items' must be a mapping")
        items = {
            str(item_id): Item(id=str(item_id), name=str(item_data["name"]))
            for item_id, item_data in raw_items.items()
        }
        return cls(
            id=str(data["id"]),
            wings=dict(data["wings"]),
            codes={str(name): str(value) for name, value in data["codes"].items()},
            items=items,
            nodes=tuple(Node.from_dict(node) for node in data["nodes"]),
        )

    def node(self, node_id: str) -> Node:
        try:
            return self._nodes_by_id[node_id]
        except KeyError as error:
            raise KeyError(f"Unknown node: {node_id}") from error

    def nodes_in_wing(self, agent: AgentId) -> tuple[Node, ...]:
        return tuple(node for node in self.nodes if node.wing == agent)

    def validate(self) -> None:
        errors: list[str] = []
        if not self.id:
            errors.append("Room id must not be empty")
        if not self.wings or set(self.wings) not in ({"A"}, {"A", "B"}):
            errors.append("Wings must contain A, and may optionally contain B")
        if any(not name for name in self.wings.values()):
            errors.append("Wing names must not be empty")

        code_values = list(self.codes.values())
        if any(len(value) != 4 or not value.isdigit() for value in code_values):
            errors.append("Every code must be a 4-digit string")
        if len(code_values) != len(set(code_values)):
            errors.append("Code values must be unique")

        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            errors.append("Node ids must be unique")
        overlap = set(node_ids) & set(self.items)
        if overlap:
            errors.append(
                "Node and item ids must not overlap: " + ", ".join(sorted(overlap))
            )

        content_parents: dict[str, list[str]] = {}
        item_locations: dict[str, list[str]] = {
            item_id: [] for item_id in self.items
        }
        for node in self.nodes:
            errors.extend(self._validate_node(node))
            for item_id in node.items:
                if item_id in item_locations:
                    item_locations[item_id].append(node.id)
            for content_id in node.contents:
                content_parents.setdefault(content_id, []).append(node.id)
                if content_id in item_locations:
                    item_locations[content_id].append(node.id)

        for node in self.nodes:
            parents = content_parents.get(node.id, [])
            if node.hidden and len(parents) != 1:
                errors.append(
                    f"Hidden node {node.id!r} must belong to exactly one container"
                )
            if parents and not node.hidden:
                errors.append(
                    f"Contained node {node.id!r} must set hidden: true"
                )

        for item_id, locations in item_locations.items():
            if len(locations) != 1:
                errors.append(
                    f"Item {item_id!r} must have exactly one location"
                )

        for agent in self.wings:
            doors = [
                node
                for node in self.nodes
                if node.wing == agent and node.kind == "door"
            ]
            if len(doors) != 1:
                errors.append(f"Wing {agent} must contain exactly one door")

        if errors:
            raise RoomValidationError("; ".join(errors))

    def _validate_node(self, node: Node) -> list[str]:
        errors: list[str] = []
        label = f"Node {node.id!r}"
        if not node.id or not node.desc:
            errors.append(f"{label} must have a non-empty id and description")
        if node.wing not in self.wings:
            errors.append(f"{label} uses undefined wing {node.wing!r}")
        if node.kind not in NODE_KINDS:
            errors.append(f"{label} has unsupported kind {node.kind!r}")

        if node.kind == "clue":
            if node.code not in self.codes:
                errors.append(f"{label} must reference a defined code")
        elif node.code is not None:
            errors.append(f"{label} may not define 'code' unless it is a clue")

        if node.kind == "spot":
            for item_id in node.items:
                if item_id not in self.items:
                    errors.append(f"{label} contains unknown item {item_id!r}")
        elif node.items:
            errors.append(f"{label} may not define 'items' unless it is a spot")

        if node.kind in {"container", "door"}:
            if node.requires is None:
                errors.append(f"{label} must define a requirement")
            else:
                errors.extend(self._validate_requirement(label, node.requires))
        elif node.requires is not None:
            errors.append(
                f"{label} may not define 'requires' unless it is a container or door"
            )

        if node.kind == "container":
            valid_contents = set(self._nodes_by_id) | set(self.items)
            for content_id in node.contents:
                if content_id not in valid_contents:
                    errors.append(f"{label} contains unknown id {content_id!r}")
        elif node.contents:
            errors.append(
                f"{label} may not define 'contents' unless it is a container"
            )
        return errors

    def _validate_requirement(
        self, node_label: str, requirement: Requirement
    ) -> list[str]:
        has_code = requirement.code is not None
        has_items = bool(requirement.items)
        if has_code == has_items:
            return [f"{node_label} must require exactly one code or item list"]
        if has_code and requirement.code not in self.codes:
            return [f"{node_label} requires unknown code {requirement.code!r}"]
        unknown_items = set(requirement.items) - set(self.items)
        if unknown_items:
            return [
                f"{node_label} requires unknown item(s): "
                + ", ".join(sorted(unknown_items))
            ]
        return []

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "wings": dict(self.wings),
            "codes": dict(self.codes),
            "items": {
                item_id: {"name": item.name} for item_id, item in self.items.items()
            },
            "nodes": [node.to_dict() for node in self.nodes],
        }


def load_room(path: str | Path) -> Room:
    """Load and validate one room from YAML."""

    with Path(path).open(encoding="utf-8") as room_file:
        data = yaml.safe_load(room_file)
    if not isinstance(data, Mapping):
        raise RoomValidationError("Room YAML must contain a mapping at its root")
    return Room.from_dict(data)


def dump_room(room: Room, path: str | Path) -> None:
    """Write a room in a deterministic, human-readable YAML representation."""

    with Path(path).open("w", encoding="utf-8") as room_file:
        yaml.safe_dump(
            room.to_dict(),
            room_file,
            sort_keys=False,
            allow_unicode=True,
        )
