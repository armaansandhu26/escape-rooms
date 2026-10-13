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
        "",
        f"YOUR WING ({engine.room.wings[agent]})",
    ]
    lines.extend(
        f"- {node.id}: {node.desc} [{_node_status(engine, agent, node)}]"
        for node in engine.visible_nodes(agent)
    )

    bag_names = [engine.room.items[item_id].name for item_id in sorted(state.bag)]
    lines.extend(["", f"YOUR BAG: {', '.join(bag_names) if bag_names else '(empty)'}"])

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
        return "open"

    assert node.requires is not None
    if node.requires.code is not None:
        return "locked: needs a 4-digit code"
    item_names = [
        engine.room.items[item_id].name for item_id in node.requires.items
    ]
    return "locked: needs " + " or ".join(item_names)
