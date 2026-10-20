"""LLM-backed action selection for one agent seat."""

from __future__ import annotations

from dataclasses import dataclass

from motiverooms.actions import ActionParseError, parse_action
from motiverooms.azure_openai import AzureResponsesClient
from motiverooms.render import render_system_prompt
from motiverooms.schema import AgentId


MECHANICS_PLAYBOOK = """
Mechanics (any framing):
- You and the other agent submit actions for the same environment tick.
- Only one take(...) per item succeeds each tick (ties favor agent A).
- Read MESSAGES and LAST RESULT before acting.
- Clue readings are private. Messages and shared notes can convey codes.
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
- Withhold codes and mislead when useful.
""",
    "mixed": """
Incentive (mixed):
- You are scored on your own escape, not the team. Share only when it helps you escape faster.
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
wait()
"""


@dataclass
class LLMAgentPolicy:
    agent: AgentId
    client: AzureResponsesClient
    framing: str = "collaborative"
    coaching: bool = False
    repair_parse: bool = False
    shared_room: bool = False
    shared_escape: str = "together"
    partner_actions: str = "private"
    mode: str = "collaborative"

    def system_prompt(self) -> str:
        base = render_system_prompt(self.framing)
        base = base.replace(
            "Escape by opening your door.",
            _shared_escape_goal(self.shared_escape) if self.shared_room else "Escape by opening your door.",
        )
        framing_playbook = FRAMING_PLAYBOOK.get(
            self.framing,
            "",
        ) if self.coaching else ""
        score_rules = {
            "collaborative": "Your reward is the sum of all agents' payoffs.",
            "mixed": "Your reward is your own payoff.",
            "competitive": "Your reward is your payoff minus the other agents' payoffs.",
        }[self.mode]
        visibility = (
            "Partner actions, including their arguments, are visible after each tick."
            if self.partner_actions == "public"
            else "Partner actions are private. Shared changes to the room remain visible."
        )
        return (
            f"You are agent {self.agent}.\n\n"
            f"{base}\n\n"
            f"{MECHANICS_PLAYBOOK}\n"
            f"{visibility}\n"
            "Round payoff: 10 plus unused turns if you escape; otherwise 0.\n"
            f"Actual scoring mode: {self.mode}. {score_rules}\n"
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
            if not self.repair_parse:
                # The engine records the parse failure and charges the turn.
                return raw, raw
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


def _shared_escape_goal(shared_escape: str) -> str:
    if shared_escape == "race":
        return (
            "Only the first agent to open the shared exit escapes; the round then ends."
        )
    if shared_escape == "together":
        return "Opening the shared exit makes every agent escape."
    raise ValueError(f"Unknown shared escape rule: {shared_escape}")
