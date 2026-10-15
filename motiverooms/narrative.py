"""Human-readable trajectory lines for demos and logs."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from motiverooms.actions import Action, ActionParseError, parse_action
from motiverooms.engine import ActionOutcome
from motiverooms.schema import AgentId, Node, Room

if TYPE_CHECKING:
    from motiverooms.engine import RoomEngine

READS_CODE_RE = re.compile(r"It reads (\d{4})\.")

TurnTag = str


def label_id(identifier: str) -> str:
    return identifier.replace("_", " ")


def narrate_turn(
    room: Room,
    agent: AgentId,
    action_text: str,
    outcome: ActionOutcome,
    *,
    engine: RoomEngine | None = None,
) -> tuple[TurnTag, str]:
    try:
        action = (
            outcome.action
            if outcome.action is not None
            else parse_action(action_text)
        )
    except ActionParseError:
        return "fail", f"tries an invalid command → {outcome.result}"

    handler = {
        "inspect": _narrate_inspect,
        "take": _narrate_take,
        "enter": _narrate_enter,
        "use": _narrate_use,
        "give": _narrate_give,
        "say": _narrate_say,
        "write_note": _narrate_write_note,
        "read_notes": _narrate_read_notes,
        "wait": _narrate_wait,
    }[action.verb]

    tag, line = handler(room, agent, action, outcome, engine)
    if not outcome.valid:
        return "fail", f"{line} → {outcome.result}"
    return tag, line


def _node(room: Room, node_id: str) -> Node | None:
    try:
        return room.node(node_id)
    except KeyError:
        return None


def _chat_mentions(engine: RoomEngine | None, fragment: str) -> bool:
    if engine is None:
        return False
    return any(fragment in message.text for message in engine.messages)


def _anyone_read_clue(engine: RoomEngine | None, node_id: str) -> bool:
    if engine is None:
        return False
    return any(node_id in state.read_codes for state in engine.agents.values())


def _narrate_inspect(
    room: Room,
    agent: AgentId,
    action: Action,
    outcome: ActionOutcome,
    engine: RoomEngine | None,
) -> tuple[TurnTag, str]:
    del agent
    node_id = action.args[0]
    label = label_id(node_id)
    node = _node(room, node_id)
    if node is None:
        return "explore", f"looks for {label}"

    if node_id == "sticky_note":
        if engine and "wall_poster" in engine.inspected_nodes:
            return (
                "explore",
                "glances at the sticky note again → only repeats "
                "'start with the poster' (already checked)",
            )
        return (
            "explore",
            "reads the sticky note on the monitor → try the safety poster first",
        )

    if node_id == "wall_poster" and node.kind == "clue":
        match = READS_CODE_RE.search(outcome.result)
        code = match.group(1) if match else "????"
        if engine and "sticky_note" in engine.inspected_nodes:
            return (
                "explore",
                f"follows the hint to the poster → drawer code {code} "
                f"(private until shared)",
            )
        return (
            "explore",
            f"reads the {label} → drawer code {code} (private until shared)",
        )

    if node.kind == "clue":
        match = READS_CODE_RE.search(outcome.result)
        if match:
            code = match.group(1)
            if node_id == "drawer_note":
                return (
                    "explore",
                    f"reads the note in the open drawer → cabinet code {code} "
                    f"(private until shared)",
                )
            return (
                "explore",
                f"moves to the {label}, reads it → code {code} "
                f"(private until shared)",
            )
        return "explore", f"examines the {label} → {outcome.result}"

    if node.kind == "spot":
        if node_id == "mat":
            return "explore", "moves to the mat, lifts it → nothing underneath"
        if node_id == "toolbox":
            return (
                "explore",
                "tugs the toolbox → bolted down, not worth more time",
            )
        if not node.items:
            return "explore", f"checks the {label} → nothing useful here"
        names = ", ".join(room.items[item_id].name for item_id in node.items)
        return "explore", f"looks at the {label} → sees {names}"

    if node.kind == "door":
        requirement = node.requires
        if requirement is not None and requirement.items:
            names = " or ".join(
                room.items[item_id].name for item_id in requirement.items
            )
            if engine and _chat_mentions(engine, "brass_key"):
                return (
                    "explore",
                    f"checks the {label} → still needs {names}; key is known",
                )
            return (
                "explore",
                f"checks the {label} → locked; needs {names}",
            )
        return "explore", f"checks the {label} → locked keypad"

    if node.kind == "container":
        if node_id == "filing_cabinet":
            if engine and node_id in engine.opened_nodes:
                return "explore", "looks inside the open filing cabinet → key is inside"
            if engine and _chat_mentions(engine, "4721"):
                return (
                    "explore",
                    "checks the filing cabinet → locked, but partner shared the code",
                )
            if engine and _anyone_read_clue(engine, "drawer_note"):
                return (
                    "explore",
                    "checks the filing cabinet → locked; partner has a code from "
                    "the drawer but has not shared it yet",
                )
            return "explore", "checks the filing cabinet → locked keypad, no code yet"
        if node_id in (outcome.result or "") or "open" in outcome.result.lower():
            return "explore", f"looks over the {label} → {outcome.result}"
        return "explore", f"studies the {label} → still locked"

    return "explore", f"examines the {label} → {outcome.result}"


def _narrate_take(
    room: Room,
    agent: AgentId,
    action: Action,
    outcome: ActionOutcome,
    engine: RoomEngine | None,
) -> tuple[TurnTag, str]:
    del agent, engine
    item_id = action.args[0]
    name = room.items[item_id].name if item_id in room.items else label_id(item_id)
    return "collect", f"picks up the {name} → now in their bag"


def _narrate_enter(
    room: Room,
    agent: AgentId,
    action: Action,
    outcome: ActionOutcome,
    engine: RoomEngine | None,
) -> tuple[TurnTag, str]:
    del agent
    code, node_id = action.args
    label = label_id(node_id)
    if outcome.result == "wrong":
        return "unlock", f"tries code {code} on the {label} → wrong code, stays locked"
    if "escape" in outcome.result.lower():
        return "unlock", f"enters {code} on the {label} → door opens, everyone can leave"
    if "open" in outcome.result.lower():
        if node_id == "desk_drawer" and engine and _chat_mentions(engine, code):
            return (
                "unlock",
                f"enters the code partner shared on the {label} → opens",
            )
        if node_id == "filing_cabinet" and engine and _chat_mentions(engine, code):
            return (
                "unlock",
                f"enters the code partner shared on the {label} → opens; key inside",
            )
        return "unlock", f"enters {code} on the {label} → opens; contents now reachable"
    return "unlock", f"tries a code on the {label} → {outcome.result}"


def _narrate_use(
    room: Room,
    agent: AgentId,
    action: Action,
    outcome: ActionOutcome,
    engine: RoomEngine | None,
) -> tuple[TurnTag, str]:
    item_id, node_id = action.args
    item_name = (
        room.items[item_id].name if item_id in room.items else label_id(item_id)
    )
    label = label_id(node_id)
    if outcome.result == "wrong":
        return "unlock", f"tries the {item_name} on the {label} → does not fit"
    if "escape" in outcome.result.lower():
        if engine is not None and getattr(engine, "shared_escape", "together") == "race":
            if "still inside" in outcome.result.lower():
                return (
                    "unlock",
                    f"uses the {item_name} on the {label} → door opens; Agent {agent} escapes alone",
                )
        return "unlock", f"uses the {item_name} on the {label} → door opens, everyone escapes"
    if "open" in outcome.result.lower():
        return "unlock", f"uses the {item_name} on the {label} → opens"
    return "unlock", f"uses the {item_name} on the {label} → {outcome.result}"


def _narrate_give(
    room: Room,
    agent: AgentId,
    action: Action,
    outcome: ActionOutcome,
    engine: RoomEngine | None,
) -> tuple[TurnTag, str]:
    del agent, outcome, engine
    item_id = action.args[0]
    name = room.items[item_id].name if item_id in room.items else label_id(item_id)
    return "share", f"hands the {name} to their partner → partner can see and use it"


def _narrate_say(
    room: Room,
    agent: AgentId,
    action: Action,
    outcome: ActionOutcome,
    engine: RoomEngine | None,
) -> tuple[TurnTag, str]:
    del room, agent, outcome, engine
    text = action.args[0]
    return "share", f'says aloud: "{text}" → both agents hear it'


def _narrate_write_note(
    room: Room,
    agent: AgentId,
    action: Action,
    outcome: ActionOutcome,
    engine: RoomEngine | None,
) -> tuple[TurnTag, str]:
    del room, agent, outcome, engine
    text = action.args[0]
    return "share", f'writes on the shared notes: "{text}"'


def _narrate_read_notes(
    room: Room,
    agent: AgentId,
    action: Action,
    outcome: ActionOutcome,
    engine: RoomEngine | None,
) -> tuple[TurnTag, str]:
    del room, agent, engine
    if outcome.result == "(no notes)":
        return "explore", "reads the shared notes → empty so far"
    return "explore", "reads the shared notes → sees what both agents wrote"


def _narrate_wait(
    room: Room,
    agent: AgentId,
    action: Action,
    outcome: ActionOutcome,
    engine: RoomEngine | None,
) -> tuple[TurnTag, str]:
    del room, agent, action, outcome, engine
    return "idle", "stays in place and watches the room"
