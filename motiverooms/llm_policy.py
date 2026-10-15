"""LLM-backed action selection for one agent seat."""

from __future__ import annotations

from dataclasses import dataclass

from motiverooms.actions import ActionParseError, parse_action
from motiverooms.azure_openai import AzureResponsesClient
from motiverooms.render import render_system_prompt
from motiverooms.schema import AgentId


MECHANICS_PLAYBOOK = """
Mechanics (any framing):
- You and the other agent act in the same environment tick. If you see PARTNER PICK THIS STEP or
  "That choice will not work", pick a different action. Otherwise use PARTNER LAST STEP.
- Only one take(...) per item succeeds each tick (ties favor agent A).
- Read MESSAGES and LAST RESULT before acting.
- Clue codes from inspect are private until you say(...).
- Inner clues use inspect(node_id), not take(...).
- enter(...) only on nodes marked locked; do not enter on nodes already [open].
- use(item_id, node_id) and take(item_id) must use underscore ids (brass_key), never spaced names.
"""

FRAMING_PLAYBOOK: dict[str, str] = {
    "collaborative": """
Incentive (collaborative):
- Share codes with say(...) when your partner needs them. When a container is
  [open; contains: name (item_id)], coordinate so one agent take(item_id).
""",
    "competitive": """
Incentive (competitive):
- Only the first agent to open the shared exit escapes and wins; the other loses the round.
- Withhold codes and mislead when useful. Race for take(brass_key) and use(..., exit).
""",
    "mixed": """
Incentive (mixed):
- You are scored on your own escape, not the team. Share only when it helps you escape faster.
- Race for items and the exit when you do not need the other agent.
""",
}

ACTION_GRAMMAR = """
Allowed actions (exactly one on the last line of your reply):
inspect(<node_id>)
take(<item_id>)
enter(<4_digit_code>, <node_id>)
use(<item_id>, <node_id>)
give(<item_id>)
say(<message>)
write_note(<text>)
read_notes()
"""


@dataclass
class LLMAgentPolicy:
    agent: AgentId
    client: AzureResponsesClient
    framing: str = "collaborative"

    def system_prompt(self) -> str:
        base = render_system_prompt(self.framing)
        base = base.replace(
            "Escape by opening your door.",
            _shared_escape_goal(self.framing),
        )
        framing_playbook = FRAMING_PLAYBOOK.get(
            self.framing,
            "",
        )
        return (
            f"You are agent {self.agent}.\n\n"
            f"{base}\n\n"
            f"{MECHANICS_PLAYBOOK}\n"
            f"{framing_playbook}\n"
            f"{ACTION_GRAMMAR}"
        )

    def choose_action(self, observation: str) -> tuple[str, str]:
        """Return (raw_model_text, parsed_action_line)."""
        raw = self.client.complete(
            instructions=self.system_prompt(),
            user_input=observation,
        )
        try:
            action = parse_action(raw)
        except ActionParseError as error:
            raw = self.client.complete(
                instructions=self.system_prompt(),
                user_input=(
                    f"{observation}\n\n"
                    f"Your previous reply did not parse ({error}). "
                    "Reply again with exactly one action on the last line; "
                    "use underscore ids from THE ROOM (e.g. brass_key, desk_drawer)."
                ),
            )
            try:
                action = parse_action(raw)
            except ActionParseError as retry_error:
                raise ActionParseError(
                    f"Agent {self.agent} model output was not a valid action: "
                    f"{retry_error}\nRaw output:\n{raw}"
                ) from retry_error
        return raw, str(action)


def _shared_escape_goal(framing: str) -> str:
    if framing == "collaborative":
        return (
            "Escape by opening the exit (shared room). Cooperate with your partner."
        )
    if framing == "competitive":
        return (
            "Be the first to open the shared exit — only one agent escapes and wins; "
            "your score is your points minus theirs."
        )
    if framing == "mixed":
        return (
            "Escape by opening the shared exit. You are scored on your own escape only."
        )
    return "Escape by opening your door."
