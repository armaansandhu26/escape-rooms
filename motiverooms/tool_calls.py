"""Room tool contracts and strict action validation, with no SDK dependencies.

This boundary knows action syntax, never room state. Visibility and feasibility
remain engine responsibilities; tool schemas must not reveal hidden objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

from motiverooms.actions import Action, ActionParseError, IDENTIFIER_RE


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: tuple[str, ...] = ()
    communication: bool = False
    navigation: bool = False
    coordination: bool = False
    large_team: bool = False

    def schema(self) -> dict:
        properties = {}
        for name in self.parameters:
            field = {"type": "string"}
            if name.endswith("_id"):
                field["description"] = "Exact identifier from your observation."
                field["pattern"] = "^[A-Za-z][A-Za-z0-9_]*$"
            elif name == "code":
                field["description"] = "Numeric code as a string, preserving leading zeros."
                field["pattern"] = "^[0-9]+$"
            else:
                field["minLength"] = 1
            properties[name] = field
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(self.parameters),
                "additionalProperties": False,
            },
        }


TOOLS = (
    ToolDefinition("inspect", "Inspect a visible object; read clues or search concealed contents.", ("node_id",)),
    ToolDefinition("take", "Pick up a visible item from a spot or opened container.", ("item_id",)),
    ToolDefinition("enter", "Try a numeric code on a locked container or door.", ("code", "node_id")),
    ToolDefinition("use", "Use an item from your bag on a container or door.", ("item_id", "node_id")),
    ToolDefinition("give", "Transfer an item from your bag to your partner.", ("item_id",), communication=True),
    ToolDefinition("say", "Send your partner a message, limited to 40 words. Costs one turn.", ("message",), communication=True),
    ToolDefinition("write_note", "Append a shared note, limited to 40 words. Costs one turn.", ("text",), communication=True),
    ToolDefinition("read_notes", "Read shared notes. Costs one turn.", communication=True),
    ToolDefinition("wait", "Spend one turn without changing the room."),
    ToolDefinition("move", "Move to an adjacent room through an open passage. Costs one turn.",
                   ("room_id",), navigation=True),
    ToolDefinition("give_to", "Transfer an item to a named teammate in your current room.",
                   ("item_id", "agent_id"), communication=True, large_team=True),
    ToolDefinition("report", "Send a private progress report to the orchestrator, limited to 40 words. Costs one turn.",
                   ("message",), communication=True, coordination=True),
)


def _available_tools(
    communication: str,
    navigation: bool,
    coordination: bool,
    team_size: int,
    worker_broadcast: bool = True,
) -> list[ToolDefinition]:
    """Share the same capability rules between discovery and call validation."""
    if communication not in {"enabled", "disabled"}:
        raise ValueError(f"Unknown communication mode: {communication}")
    available = []
    for tool in TOOLS:
        if tool.communication and communication == "disabled":
            continue
        if tool.navigation and not navigation:
            continue
        if tool.coordination and not coordination:
            continue
        if tool.name == "say" and not worker_broadcast:
            continue
        # Small teams have an implicit recipient; larger teams must name one.
        if tool.large_team and team_size <= 2:
            continue
        if tool.name == "give" and team_size > 2:
            continue
        # A solo worker can report to a controller but has no teammates.
        if team_size == 1 and tool.communication and tool.name != "report":
            continue
        available.append(tool)
    return available


def tool_definitions(*, communication: str = "enabled", navigation: bool = False,
                     coordination: bool = False, team_size: int = 2,
                     worker_broadcast: bool = True) -> list[dict]:
    return [tool.schema() for tool in _available_tools(communication, navigation, coordination, team_size, worker_broadcast)]


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: str

    def json_arguments(self) -> dict:
        """Decode one JSON object, rejecting duplicate keys and non-object values."""
        if not isinstance(self.arguments, str):
            raise ActionParseError("Tool arguments must be a JSON string")
        try:
            arguments = json.loads(self.arguments, object_pairs_hook=_unique_object)
        except (ValueError, TypeError) as error:
            raise ActionParseError("Tool arguments must be a valid JSON object") from error
        if not isinstance(arguments, dict):
            raise ActionParseError("Tool arguments must be a JSON object")
        return arguments

    def action(self, *, communication: str = "enabled", navigation: bool = False,
               coordination: bool = False, team_size: int = 2, worker_broadcast: bool = True) -> Action:
        allowed = {tool.name: tool for tool in _available_tools(communication, navigation, coordination, team_size, worker_broadcast)}
        if not isinstance(self.name, str) or self.name not in allowed:
            raise ActionParseError(f"Unavailable tool: {self.name}")
        arguments = self.json_arguments()
        definition = allowed[self.name]
        if set(arguments) != set(definition.parameters):
            raise ActionParseError(f"{self.name} requires exactly: {', '.join(definition.parameters) or 'no arguments'}")
        values = []
        for key in definition.parameters:
            value = arguments[key]
            if not isinstance(value, str) or not value.strip():
                raise ActionParseError(f"{key} must be a nonempty string")
            if key.endswith("_id") and not IDENTIFIER_RE.fullmatch(value):
                raise ActionParseError(f"{key} must be an exact identifier")
            if key == "code" and (not value.isascii() or not value.isdigit()):
                raise ActionParseError("code must be a numeric string")
            # The engine's message limit is whitespace-based. Normalize here so
            # multiline text safely round-trips through the legacy action format.
            values.append(" ".join(value.split()) if key in {"message", "text"} else value)
        return Action(self.name, tuple(values))


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate argument: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class ToolResponse:
    """Serializable native output items, including reasoning needed on continuation."""

    output: tuple[dict, ...]
    text: str = ""
    status: str | None = None

    @property
    def calls(self) -> tuple[ToolCall, ...]:
        return tuple(ToolCall(item.get("call_id", ""), item.get("name", ""), item.get("arguments", ""))
                     for item in self.output if item.get("type") == "function_call")

    def single_call(self) -> ToolCall:
        if self.status not in {None, "completed"}:
            raise ActionParseError(f"Tool response did not complete: {self.status}")
        calls = self.calls
        if len(calls) != 1 or not isinstance(calls[0].call_id, str) or not calls[0].call_id:
            raise ActionParseError("Exactly one tool call with a call_id is required per turn")
        return calls[0]

    def single_action(self, *, communication: str = "enabled", navigation: bool = False,
                      coordination: bool = False, team_size: int = 2, worker_broadcast: bool = True) -> Action:
        return self.single_call().action(communication=communication, navigation=navigation,
                                         coordination=coordination, team_size=team_size, worker_broadcast=worker_broadcast)
