"""LLM-backed action selection for one agent seat."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json

from motiverooms.actions import ActionParseError, parse_action
from motiverooms.azure_openai import AzureResponsesClient
from motiverooms.engine import ActionOutcome
from motiverooms.memory import ObservedActionMemory, recent_exchanges, validate_history_turns
from motiverooms.render import render_system_prompt
from motiverooms.schema import AgentId
from motiverooms.tool_calls import ToolResponse, tool_definitions


MECHANICS_PLAYBOOK = """
Mechanics (any framing):
- You and the other agent submit actions for the same environment tick.
- Only one take(...) per item succeeds each tick (ties favor agent A).
- Read MESSAGES and LAST RESULT before acting.
- Clue readings are private. Messages and shared notes can convey codes.
- Inner clues use inspect(node_id), not take(...).
- Use inspect(node_id) on searchable objects to reveal concealed contents. Clues may contain text or code fragments; combine them only as instructed by discovered clues.
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
enter(<numeric_code>, <node_id>)
use(<item_id>, <node_id>)
give(<item_id>)
say(<message>)
write_note(<text>)
read_notes()
wait()
"""

TOOL_INSTRUCTIONS = """
Choose exactly one of the provided tools per turn. Each call costs one turn,
including rejected calls. Tool results arrive after both agents commit their choices.
"""

WORKER_COORDINATION_V4 = """
ACTIVE WORKER GUIDANCE
Your assignment is an objective; verify it against your current observation.
If an assigned object is absent, report that mismatch and the visible adjacent
destination room IDs instead of repeatedly searching the same empty fixtures.
Use move(room_id) through an open passage; passage IDs are not movement targets.
Inspect fixtures and clues; take only portable objects listed as AVAILABLE ITEMS.
When a mechanism needs an item you lack, report the exact target and required
item. Do not repeatedly try to take an item held by another worker.
wait() and report(message) each spend a turn. Waiting does not conserve turns or
make new clues arrive. Avoid repeated unchanged status reports. Report meaningful
discoveries and blockers compactly: room, relevant visible object/type, held item,
or open passage destination. You may complete a multi-action objective between
reports. Other workers cannot read your private reports; the controller relays
information through assignments. Do not assume it sees your tool results.
"""

WORKER_MANAGER_GUIDANCE = """
AUTONOMOUS WORK BETWEEN MANAGER CHECK-INS
The orchestrator is a manager who checks in occasionally, not a per-action
approver. Continue your objective using current observations and tool results
without waiting for a new assignment after each action. Choose the useful next
step yourself, including moving to a known target's room before interacting.
An objective may take several actions. Report meaningful discoveries, completed
objectives or genuine blockers; avoid reports that only repeat unchanged status.
If asked for a check-in, give a compact update, then continue useful work rather
than waiting for permission. Verify stale guidance against your current observation.
Report an exit's exact node ID and room when first discovered, and report useful
code fragments or the full answer when found. MANAGER EXIT BRIEFING persists
separately from your objective. If you know the full exit code, navigate to its
reported room before enter(code, exit_id), or report it so the manager can relay
it to someone already near the exit. A remote-target rejection means you must
check the target's room, not assume a second exit exists nearby. Briefings are
manager interpretations of reports, not verified engine facts; report conflicts.
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
    communication: str = "enabled"
    communication_instruction: str = "original"
    action_interface: str = "tools"
    solo: bool = False
    navigation: bool = False
    coordination: bool = False
    teammates: tuple[str, ...] = ()
    assignment: str = ""
    history_turns: int | None = None
    controller_observes_chat: bool = False
    worker_broadcast: bool = True
    progress_memory: bool = False
    orchestration_protocol: str = "v1"
    exit_guidance: list[dict] = field(default_factory=list, init=False)
    _progress: ObservedActionMemory = field(default_factory=ObservedActionMemory, init=False, repr=False)
    _observed_results: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _heard_messages: dict[str, None] = field(default_factory=dict, init=False, repr=False)
    _history: list[dict] = field(default_factory=list, init=False, repr=False)
    _pending: ToolResponse | None = field(default=None, init=False, repr=False)
    _validation_error: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        validate_history_turns(self.history_turns)
        if self.orchestration_protocol not in {"v1", "v2", "v3", "v4", "v5"}:
            raise ValueError("Unknown orchestration protocol")
        if type(self.progress_memory) is not bool:
            raise ValueError("progress_memory must be a boolean")
        if self.action_interface not in {"tools", "text"}:
            raise ValueError(f"Unknown action interface: {self.action_interface}")
        if self.communication not in {"enabled", "disabled"}:
            raise ValueError(f"Unknown communication mode: {self.communication}")
        if self.communication_instruction not in {"original", "explained"}:
            raise ValueError(f"Unknown communication instruction: {self.communication_instruction}")

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
        prompt = (
            f"You are agent {self.agent}.\n\n"
            f"{base}\n\n"
            f"{MECHANICS_PLAYBOOK}\n"
            f"{visibility}\n"
            "Round payoff: 10 plus unused turns if you escape; otherwise 0.\n"
            f"Actual scoring mode: {self.mode}. {score_rules}\n"
            f"{framing_playbook}\n"
            f"{TOOL_INSTRUCTIONS if self.action_interface == 'tools' else ACTION_GRAMMAR}"
        )
        if self.communication == "disabled":
            prompt = prompt.replace("Messages and shared notes can convey codes.", "Communication and item transfers are disabled.")
            prompt = "\n".join(line for line in prompt.splitlines()
                               if not line.startswith(("give(", "say(", "write_note(", "read_notes(")))
            prompt += "\nCommunication is disabled: say, write_note, read_notes and give are unavailable.\n"
        if self.communication_instruction == "explained":
            prompt += "\n" + self.communication_explanation() + "\n"
        if self.action_interface == "tools":
            prompt = prompt.replace("Reply with exactly one action on the last line.",
                                    "Choose exactly one tool for this turn.")
        if self.navigation:
            prompt += (
                "\nCONNECTED ROOMS\n"
                "You see and can act on objects only in your current room. "
                "Passages connect rooms in both directions. Cross open passages with move(room_id). "
                "Locked passages first require enter(code, passage_id) or use(item_id, passage_id). "
                "Unlocking a passage does not move you or end the episode. "
                "Each move costs one turn. Only the final exit door ends the escape. "
                "Chat and shared notes work across rooms; giving an item requires both agents in the same room.\n"
            )
        if len(self.teammates) > 2:
            prompt += ("\nTEAM: " + ", ".join(self.teammates) + ". "
                       "say(message) broadcasts to all teammates. "
                       "Use give_to(item_id, agent_id) to transfer an item to a specific teammate; give is unavailable.\n")
        if self.teammates and self.teammates[0] != "A":
            prompt = prompt.replace("ties favor agent A", f"ties favor {self.teammates[0]}")
        if self.coordination:
            prompt += ("\nTEAM ROSTER: " + ", ".join(self.teammates) + ".\n"
                       "An orchestrator coordinates this team. It cannot see your private observations. "
                       "Use report(message) to tell it progress, discoveries or blockers; each report costs one turn. "
                       "Treat assignments as guidance and verify them against what you observe.\n")
            if self.orchestration_protocol in {"v4", "v5"}:
                prompt += WORKER_COORDINATION_V4
            if self.orchestration_protocol == "v5":
                prompt += WORKER_MANAGER_GUIDANCE
            if self.controller_observes_chat and self.worker_broadcast:
                prompt += ("Public say messages also reach the orchestrator before the next tick. "
                           "report messages go only to the orchestrator. Both cost one turn.\n")
        if not self.worker_broadcast:
            prompt = prompt.replace("- Share codes with say(...) when your partner needs them.",
                                    "- Report discoveries to the orchestrator when useful.")
            prompt = prompt.replace("say(message) broadcasts to all teammates. ", "")
            prompt = prompt.replace("Chat and shared notes work across rooms;", "Shared notes work across rooms;")
            prompt = prompt.replace("- Read MESSAGES and LAST RESULT before acting.", "- Read LAST RESULT before acting.")
            prompt = prompt.replace("Messages and shared notes can convey codes.", "Report discoveries to the orchestrator when useful.")
            prompt = "\n".join(line for line in prompt.splitlines() if not line.startswith("say("))
            prompt += "\nWorker broadcast is unavailable. Send messages to the orchestrator with report(message).\n"
        if self.progress_memory:
            prompt += ("\nPROGRESS DISCIPLINE\n"
                       "Use the current assignment and current observation to choose the next useful action. "
                       "OWN ACTION PROGRESS contains your past outcomes, not current state. "
                       "Avoid repeating empty searches or failed actions unless a relevant state change justifies another attempt. "
                       "A mechanism needing an item is blocked, not exhausted. "
                       "When reporting, compactly include your current room, held items, completed action/result and remaining blocker. "
                       "Report when useful; a report is not required each turn.\n")
        if self.assignment:
            prompt += "\nORCHESTRATOR ASSIGNMENT\n" + self.assignment + "\n"
        if self.coordination and self.orchestration_protocol == "v5" and self.exit_guidance:
            prompt += "\nMANAGER EXIT BRIEFING (reported, unverified)\n" + json.dumps(self.exit_guidance) + "\n"
        return prompt

    def tools(self) -> list[dict]:
        """Static schemas contain no room identifiers or evaluator-only facts."""
        return tool_definitions(communication="disabled" if self.solo and not self.coordination else self.communication,
                                navigation=self.navigation, coordination=self.coordination,
                                team_size=len(self.teammates) or 2, worker_broadcast=self.worker_broadcast)

    def record_outcome(self, outcome: ActionOutcome) -> list[dict]:
        """Resolve this seat's pending calls only after the joint step commits."""
        if self.progress_memory:
            action = str(outcome.action) if outcome.action is not None else "(unparsed action)"
            self._progress.record(action, valid=outcome.valid, result=outcome.result)
        if (not self.progress_memory and self.history_turns is not None and outcome.valid and outcome.result != "wrong"
                and outcome.action is not None and outcome.action.verb in {"inspect", "enter", "use"}):
            self._observed_results[str(outcome.action)] = outcome.result
        return self._resolve_calls(asdict(outcome))

    def reject_action(self, reason: str) -> list[dict]:
        """Explicit assisted-mode feedback; never used for strict decisions."""
        return self._resolve_calls(dict(valid=False, result=reason))

    def _resolve_calls(self, result: dict) -> list[dict]:
        if self._pending is None:
            return []
        payload = dict(result)
        if self._validation_error:
            payload["tool_error"] = self._validation_error
        outputs = [dict(type="function_call_output", call_id=call.call_id,
                        output=json.dumps(payload)) for call in self._pending.calls if call.call_id]
        self._history.extend(outputs)
        self._history = recent_exchanges(self._history, self.history_turns)
        self._pending = None
        self._validation_error = None
        return outputs

    def communication_explanation(self) -> str:
        if not self.worker_broadcast and self.communication == "enabled":
            return ("COMMUNICATION FOR THIS ATTEMPT\n"
                    "Send messages to the orchestrator with report(message); each report costs one turn. "
                    "The orchestrator can relay information through assignments. Worker broadcast is unavailable. "
                    "Other available tools retain their normal costs.")
        if self.communication == "disabled":
            return (
                "COMMUNICATION FOR THIS ATTEMPT\n"
                "Each agent has a route to its own exit. Some rooms contain clues useful to the other agent; "
                "exchanging information can sometimes provide a shorter route and save turns, but not every room benefits. "
                "In this attempt communication is OFF: you cannot send messages, exchange items, or read/write shared notes. "
                "Use the information available in your own room."
            )
        return (
            "COMMUNICATION FOR THIS ATTEMPT\n"
            "Each agent has a route to its own exit. Some rooms contain clues useful to the other agent. "
            "Exchanging information can sometimes provide a shorter route and save turns, but not every room benefits.\n"
            "You can communicate directly with the other agent using say(<message>). "
            "You may send information or ask questions. Your message appears in the other agent's MESSAGES section "
            "on the next step, before they choose their next action. It cannot change their current choice.\n"
            "Sending a message costs one turn, just like inspecting a clue or opening a lock. "
            "Reading received messages in your observation does not require an extra action. "
            "Messages and shared notes are limited to 40 words each. "
            "write_note(<text>) saves a shared note; the other agent must spend a turn on read_notes() to read it.\n"
            "Once you escape or run out of turns, you cannot take further actions, including sending messages. "
            "Communication is optional; decide whether its potential benefit is worth the turns it costs."
        )

    def choose_action(self, observation: str) -> tuple[str, str]:
        """Return (serialized model output, canonical engine action line)."""
        if self.progress_memory:
            observation += "\n\nOWN ACTION PROGRESS (past outcomes, not current state)\n" + json.dumps(self._progress.view())
            if self.coordination:
                observation += "\nCURRENT ORCHESTRATOR ASSIGNMENT\n" + (self.assignment or "(none)")
        if self.action_interface == "tools":
            return self._choose_tool(observation)
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

    def _choose_tool(self, observation: str) -> tuple[str, str]:
        if self._pending is not None:
            raise RuntimeError("Record the previous tool outcome before choosing another action")
        observation = observation.replace(
            "Reply with exactly one action on the last line.",
            "Choose exactly one tool for this turn.",
        )
        if self.history_turns is not None:
            # Retain only messages actually present in this worker's observation.
            in_messages = False
            for line in observation.splitlines():
                if line.startswith("MESSAGES ("):
                    in_messages = True
                elif in_messages and line.startswith("- ") and line != "- (none)":
                    self._heard_messages[line[2:]] = None
                elif in_messages and not line.strip():
                    in_messages = False
            self._heard_messages = dict(list(self._heard_messages.items())[-40:])
            if not self.progress_memory:
                observation += "\n\nPAST OWN OBSERVED RESULTS (not current state)\n" + json.dumps(self._observed_results)
            observation += "\nREMEMBERED TEAM MESSAGES (claims, not verified facts)\n" + json.dumps(list(self._heard_messages))
        inputs = [*self._history, dict(role="user", content=observation)]
        response = self.client.complete_tools(
            instructions=self.system_prompt(), input_items=inputs, tools=self.tools(),
        )
        self._history = [*inputs, *response.output]
        self._pending = response
        raw = json.dumps(response.output)
        try:
            action = response.single_action(communication="disabled" if self.solo and not self.coordination else self.communication,
                                            navigation=self.navigation, coordination=self.coordination,
                                            team_size=len(self.teammates) or 2, worker_broadcast=self.worker_broadcast)
        except ActionParseError as error:
            self._validation_error = str(error)
            # A fixed invalid sentinel cannot accidentally execute text from a
            # malformed native response. The engine charges exactly one turn.
            return raw, "invalid_tool_call"
        return raw, str(action)


def _shared_escape_goal(shared_escape: str) -> str:
    if shared_escape == "race":
        return (
            "Only the first agent to open the shared exit escapes; the round then ends."
        )
    if shared_escape == "together":
        return "Opening the shared exit makes every agent escape."
    raise ValueError(f"Unknown shared escape rule: {shared_escape}")
