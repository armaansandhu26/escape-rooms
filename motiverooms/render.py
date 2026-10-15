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


def render_observation(
    engine: RoomEngine,
    agent: AgentId,
    *,
    round_number: int = 1,
    total_rounds: int = 1,
    message_limit: int = 6,
    partner_action_steps: int = 3,
) -> str:
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
                "HINT: Codes you read from clues are private until you say(...) them.",
                "If a container is [open; contains: ...], use take(<item_id>) on that item.",
            ]
        )
    lines.extend(
        [
            "",
            (
                f"THE ROOM ({engine.room.space})"
                if engine.room.is_shared
                else f"YOUR WING ({engine.room.wings[agent]})"
            ),
        ]
    )
    lines.extend(
        f"- {node.id}: {node.desc} [{_node_status(engine, agent, node)}]"
        for node in engine.visible_nodes(agent)
    )

    bag_names = [engine.room.items[item_id].name for item_id in sorted(state.bag)]
    lines.extend(["", f"YOUR BAG: {', '.join(bag_names) if bag_names else '(empty)'}"])

    if partner is not None:
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


def _node_status(engine: RoomEngine, agent: AgentId, node: Node) -> str:
    state = engine.agents[agent]
    if node.kind == "clue":
        code = state.read_codes.get(node.id)
        return f"read: code {code}" if code is not None else "not read"
    if node.kind == "spot":
        available = [
            engine.room.items[item_id].name
            for item_id in node.items
            if item_id not in engine.taken_items
        ]
        return "contains: " + ", ".join(available) if available else "empty"
    if node.id in engine.opened_nodes:
        if node.kind == "container":
            available = _open_container_labels(engine, agent, node)
            if available:
                return "open; contains: " + ", ".join(available)
        return "open"

    assert node.requires is not None
    if node.requires.code is not None:
        return "locked: needs a 4-digit code"
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
