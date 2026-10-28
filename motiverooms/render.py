"""Human-readable observations and incentive framing."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml

from motiverooms.engine import RoomEngine
from motiverooms.schema import AgentId, Node

DEFAULT_FRAMING_PATH = (
    Path(__file__).parents[1] / "rooms" / "templates" / "framing.yaml"
)


def load_framing_prompts(path: str | Path = DEFAULT_FRAMING_PATH) -> dict[str, str]:
    with Path(path).open(encoding="utf-8") as prompt_file:
        prompts = yaml.safe_load(prompt_file)
    if not isinstance(prompts, Mapping) or not all(
        isinstance(name, str) and isinstance(text, str)
        for name, text in prompts.items()
    ):
        raise ValueError("Framing template must map names to prompt strings")
    return dict(prompts)


FRAMING_PROMPTS = load_framing_prompts()


def render_system_prompt(framing: str) -> str:
    try:
        incentive = FRAMING_PROMPTS[framing]
    except KeyError as error:
        choices = ", ".join(FRAMING_PROMPTS)
        raise ValueError(f"Unknown framing {framing!r}; choose from {choices}") from error
    return (
        f"{incentive}\n\n"
        "Escape by opening your door. Objects do exactly what their descriptions "
        "say. Reply with exactly one action on the last line."
    )


def _render_legacy_observation(
    engine: RoomEngine,
    agent: AgentId,
    *,
    round_number: int = 1,
    total_rounds: int = 1,
    message_limit: int = 6,
    partner_action_steps: int = 3,
    partner_actions: str = "private",
) -> str:
    if partner_actions not in {"private", "public"}:
        raise ValueError(f"Unknown partner action visibility: {partner_actions}")
    state = engine.agents[agent]
    partner = next((other for other in engine.agents if other != agent), None)
    if partner is None:
        partner_status = "none"
    elif engine.agents[partner].escaped:
        partner_status = "escaped"
    elif engine.agents[partner].turns_left == 0:
        partner_status = "out of turns"
    else:
        partner_status = "still in the room"
    if len(engine.agents) > 2:
        partner_status = "; ".join(
            f"{other}: {'escaped' if state.escaped else 'out of turns' if state.turns_left <= 0 else 'active'}"
            for other, state in engine.agents.items() if other != agent
        )

    lines = [
        (
            f"ROUND {round_number}/{total_rounds} | You are agent {agent} | "
            f"Turns left: {state.turns_left} | Partner: {partner_status}"
        ),
    ]
    if partner is not None:
        lines.extend(
            [
                "",
                (
                    "Clue readings are private. Partner action arguments are public after each tick."
                    if partner_actions == "public"
                    else "Clue readings and partner actions are private; messages and shared notes can convey codes."
                ),
                "If a container is [open; contains: ...], use take(<item_id>) on that item.",
            ]
        )
    lines.extend(
        [
            "",
            (
                f"CURRENT ROOM: {state.location} ({engine.room.locations[state.location]})"
                if engine.room.locations else f"THE ROOM ({engine.room.space})"
                if engine.room.is_shared
                else f"YOUR WING ({engine.room.wings[agent]})"
            ),
        ]
    )
    if engine.room.locations:
        lines += ["", "PASSAGES (unlock a locked passage with enter/use, then move to the room_id)"]
        for passage in engine.visible_passages(agent):
            target = passage.other_location(state.location)
            status = "open" if engine.passage_is_open(passage) else _node_status(engine, agent, passage)
            lines.append(f"- {passage.id}: {passage.desc} [{status}]; room_id={target} ({engine.room.locations[target]})")
        lines += ["", "OBJECTS IN YOUR CURRENT ROOM"]
    lines.extend(
        f"- {node.id}: {node.desc} [{_node_status(engine, agent, node)}]"
        for node in engine.visible_nodes(agent)
    )

    bag_names = [engine.room.items[item_id].name for item_id in sorted(state.bag)]
    lines.extend(["", f"YOUR BAG: {', '.join(bag_names) if bag_names else '(empty)'}"])
    if engine.room.locations:
        clue_heading = "YOUR READ CLUES" if any(node.text is not None and node.kind == "clue" for node in engine.room.nodes) else "YOUR READ CODES"
        lines += ["", clue_heading, *(
            [f"- {node_id}: {code}" for node_id, code in sorted(state.read_codes.items())] or ["- (none)"]
        )]

    if len(engine.agents) > 2 and partner_actions == "public":
        for other in engine.agents:
            if other == agent:
                continue
            lines += ["", f"TEAMMATE {other} RECENT ACTIONS"]
            lines += [f"- {action}" for action in _partner_action_history(engine, other, partner_action_steps)]
    elif partner is not None and partner_actions == "public":
        lines.extend(["", "PARTNER LAST STEP"])
        partner_last = engine.last_joint_actions.get(partner)
        lines.append(f"- {partner_last}" if partner_last else "- (none yet)")

        if partner_action_steps > 0:
            recent_partner = _partner_action_history(
                engine, partner, partner_action_steps
            )
            lines.extend(["", f"PARTNER RECENT (last {partner_action_steps})"])
            if recent_partner:
                lines.extend(f"- {action}" for action in recent_partner)
            else:
                lines.append("- (none yet)")

    if partner is not None:
        lines.extend(["", f"MESSAGES (last {message_limit})"])
        recent_messages = engine.messages[-message_limit:]
        if recent_messages:
            lines.extend(
                f"- {message.sender}: {message.text}" for message in recent_messages
            )
        else:
            lines.append("- (none)")

    lines.extend(
        [
            "",
            f"LAST RESULT: {state.last_result}",
            "",
            "Reply with exactly one action on the last line.",
        ]
    )
    return "\n".join(lines)


def render_observation(
    engine: RoomEngine,
    agent: AgentId,
    *,
    round_number: int = 1,
    total_rounds: int = 1,
    message_limit: int = 6,
    partner_action_steps: int = 3,
    partner_actions: str = "private",
    observation_version: str = "v1",
) -> str:
    """Versioned presentation of the same observable state; v1 preserves old logs."""
    if observation_version not in {"v1", "v2", "v3"}:
        raise ValueError(f"Unknown observation version: {observation_version}")
    original = _render_legacy_observation(
        engine, agent, round_number=round_number, total_rounds=total_rounds,
        message_limit=message_limit, partner_action_steps=partner_action_steps,
        partner_actions=partner_actions,
    )
    if observation_version == "v1":
        return original
    visible = {node.id: node for node in engine.visible_nodes(agent)}
    lines = []
    if observation_version == "v3":
        code_argument = "4_digit_code" if all(len(code) == 4 for code in engine.room.codes.values()) else "numeric_code"
        lines += [
            "LOCK AND ID RULES",
            ("- CODE LOCK: enter(<4_digit_code>, <node_id>) uses a code read from a clue or received in a message."
             if code_argument == "4_digit_code" else
             "- CODE LOCK: enter(<numeric_code>, <node_id>) uses discovered clues; some puzzles require combining fragments."),
            "- ITEM LOCK: use(<item_id>, <node_id>) uses a physical item already in YOUR BAG; enter(...) cannot open an item lock.",
            "- Digits inside object/item IDs are identifier text, not evidence of a lock code. Do not infer a code from an ID's numeric suffix.",
            "",
        ]
    for line in original.splitlines():
        if line.startswith("- ") and ": " in line:
            node_id = line[2:].split(": ", 1)[0]
            node = visible.get(node_id)
            if node is not None:
                line = line.replace(f"- {node_id}:", f"- {node_id} (type={node.kind}):", 1)
                if node.kind == "clue":
                    line += f" Read with inspect({node_id}); this is not a takeable item."
                elif node.kind == "search":
                    line += f" Search with inspect({node_id}) to reveal anything concealed here."
                elif node.kind in {"container", "door"}:
                    if node_id in engine.opened_nodes:
                        line += " STATUS: OPEN. Unlocking is complete; enter/use on this node again will fail."
                    elif node.requires and node.requires.items:
                        line += " Accepted item IDs: " + ", ".join(node.requires.items) + "."
                    if observation_version == "v3" and node_id not in engine.opened_nodes and node.requires:
                        if node.requires.code is not None:
                            line += (f" CODE LOCK: enter(<4_digit_code>, {node_id}); use a code obtained from a clue/message."
                                     if code_argument == "4_digit_code" else
                                     f" CODE LOCK: enter(<numeric_code>, {node_id}); use discovered clues.")
                        else:
                            line += f" ITEM LOCK: use(<item_id>, {node_id}); carry an accepted item first. This lock does not accept enter(...)."
        if line.startswith("YOUR BAG:"):
            labels = [f"{engine.room.items[i].name} (item_id={i})" for i in sorted(engine.agents[agent].bag)]
            line = "YOUR BAG: " + (", ".join(labels) if labels else "(empty)")
            lines.append(line)
            available = []
            for node in visible.values():
                contents = node.items if node.kind == "spot" else (
                    node.contents if node.kind in {"container", "search"} and node.id in engine.opened_nodes else ()
                )
                available.extend(f"- {i} (item; source={node.id})" for i in contents
                                 if i in engine.room.items and i not in engine.taken_items)
            lines += ["", "AVAILABLE ITEMS (takeable objects only)", *(available or ["- (none)"])]
            continue
        if line.startswith("LAST RESULT:"):
            recent = [step[agent] for step in engine.joint_action_history if agent in step][-3:]
            lines += ["YOUR RECENT ACTIONS (last 3)", *[f"- {a}" for a in recent]] if recent else []
            lines.append(f"YOUR LAST ACTION: {recent[-1] if recent else '(none yet)'}")
        lines.append(line)
    return "\n".join(lines)


def _node_status(engine: RoomEngine, agent: AgentId, node: Node) -> str:
    state = engine.agents[agent]
    if node.kind == "clue":
        code = state.read_codes.get(node.id)
        return (f"read: {code}" if node.text is not None else f"read: code {code}") if code is not None else "not read"
    if node.kind == "spot":
        available = [
            engine.room.items[item_id].name
            for item_id in node.items
            if item_id not in engine.taken_items
        ]
        return "contains: " + ", ".join(available) if available else "empty"
    if node.kind == "scenery":
        return "inspect for details"
    if node.kind == "search" and node.id not in engine.opened_nodes:
        return "not searched"
    if node.id in engine.opened_nodes:
        if node.kind in {"container", "search"}:
            available = _open_container_labels(engine, agent, node)
            if available:
                return "open; contains: " + ", ".join(available)
        return "open"

    assert node.requires is not None
    if node.requires.code is not None:
        return f"locked: needs a {len(engine.room.codes[node.requires.code])}-digit code"
    item_names = [
        engine.room.items[item_id].name for item_id in node.requires.items
    ]
    return "locked: needs " + " or ".join(item_names)


def _open_container_labels(
    engine: RoomEngine, agent: AgentId, container: Node
) -> list[str]:
    """Container contents may be item ids or inner node ids (e.g. clues)."""
    labels: list[str] = []
    for content_id in container.contents:
        label = _format_content_id(engine, agent, content_id)
        if label is not None:
            labels.append(label)
    return labels


def _format_content_id(
    engine: RoomEngine, agent: AgentId, content_id: str
) -> str | None:
    if content_id in engine.room.items:
        if content_id in engine.taken_items:
            return None
        item = engine.room.items[content_id]
        return f"{item.name} ({content_id})"

    try:
        inner = engine.room.node(content_id)
    except KeyError:
        return None
    if not engine.is_visible(inner):
        return None
    if inner.kind == "clue":
        if content_id in engine.agents[agent].read_codes:
            return None
        return f"{content_id} (inspect clue)"
    return content_id


def _partner_action_history(
    engine: RoomEngine, partner: AgentId, limit: int
) -> list[str]:
    collected: list[str] = []
    for step_actions in reversed(engine.joint_action_history):
        action = step_actions.get(partner)
        if action is None:
            continue
        collected.append(action)
        if len(collected) >= limit:
            break
    collected.reverse()
    return collected
