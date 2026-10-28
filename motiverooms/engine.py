"""Deterministic state transitions for one MotiveRooms room."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from collections.abc import Mapping

from motiverooms.actions import Action, ActionParseError, parse_action
from motiverooms.joint_step import TAKE_CONFLICT_MESSAGE, VERB_PHASES
from motiverooms.schema import AgentId, Node, Passage, Room

SharedEscapeMode = Literal["together", "race"]


@dataclass
class AgentState:
    turns_left: int
    bag: set[str] = field(default_factory=set)
    read_codes: dict[str, str] = field(default_factory=dict)
    escaped: bool = False
    last_result: str = "The round has not started."
    location: str | None = None


@dataclass(frozen=True)
class Message:
    sender: AgentId
    text: str


@dataclass(frozen=True)
class ActionOutcome:
    action: Action | None
    valid: bool
    result: str


class RoomEngine:
    """Deterministic room state.

    ``step`` applies one agent action (sequential API, tests and split-wing play).
    ``step_joint`` is one environment tick: every still-active agent submits an
    action; all are applied in verb phases with deterministic tie-break order
    (A before B). Each acting agent spends one turn from their budget per tick.
    """

    def __init__(
        self,
        room: Room,
        turn_budget: int,
        *,
        max_message_words: int = 40,
        shared_escape: SharedEscapeMode = "together",
        communication: str = "enabled",
        coordination: bool = False,
        worker_broadcast: bool = True,
    ) -> None:
        if turn_budget <= 0:
            raise ValueError("turn_budget must be positive")
        self.room = room
        self.max_message_words = max_message_words
        self.shared_escape = shared_escape
        if communication not in {"enabled", "disabled"}:
            raise ValueError(f"Unknown communication mode: {communication}")
        self.communication = communication
        self.coordination = coordination
        if type(worker_broadcast) is not bool:
            raise ValueError("worker_broadcast must be a boolean")
        self.worker_broadcast = worker_broadcast
        self.agents = {
            agent: AgentState(turns_left=turn_budget, location=room.start_locations.get(agent))
            for agent in room.wings
        }
        self.opened_nodes: set[str] = set()
        self.taken_items: set[str] = set()
        self.messages: list[Message] = []
        self.reports: list[Message] = []
        self.notes: list[str] = []
        self.inspected_nodes: set[str] = set()
        self.last_joint_actions: dict[AgentId, str] = {}
        self.joint_action_history: list[dict[AgentId, str]] = []
        self._joint_history_limit = 16

    def is_visible(self, node: Node) -> bool:
        if not node.hidden:
            return True
        return any(
            parent.id in self.opened_nodes and node.id in parent.contents
            for parent in self.room.nodes
            if parent.kind in {"container", "search"}
        )

    def visible_nodes(self, agent: AgentId) -> tuple[Node, ...]:
        return tuple(
            node
            for node in self.room.nodes_for_agent(agent)
            if self.is_visible(node) and (
                not self.room.locations or node.location == self.agents[agent].location
            )
        )

    def visible_passages(self, agent: AgentId) -> tuple[Passage, ...]:
        location = self.agents[agent].location
        return tuple(passage for passage in self.room.passages if location in passage.locations)

    def passage_is_open(self, passage: Passage) -> bool:
        return passage.requires is None or passage.id in self.opened_nodes

    def step(self, agent: AgentId, output: str | Action) -> ActionOutcome:
        state = self._agent_state(agent)
        if state.escaped:
            return ActionOutcome(None, False, "You have already escaped.")
        if state.turns_left <= 0:
            return ActionOutcome(None, False, "You have no turns left.")

        state.turns_left -= 1
        raw_line = output if isinstance(output, str) else str(output)
        try:
            action = parse_action(output) if isinstance(output, str) else output
        except ActionParseError as error:
            outcome = self._finish(agent, None, False, f"Invalid action: {error}")
            self._record_joint_step({agent: raw_line})
            return outcome

        outcome = self._apply_parsed_action(agent, action)
        self._record_joint_step({agent: raw_line})
        return outcome

    def step_joint(
        self, actions: Mapping[AgentId, str | Action]
    ) -> dict[AgentId, ActionOutcome]:
        order = tuple(self.room.wings)
        active = [agent for agent in order if self._can_act(agent)]
        if not active:
            raise ValueError("The round is over; no agent can act.")

        missing = [agent for agent in active if agent not in actions]
        if missing:
            agents = ", ".join(missing)
            raise ValueError(f"Missing action(s) for active agent(s): {agents}")

        for agent in active:
            self.agents[agent].turns_left -= 1

        parsed: dict[AgentId, Action | None] = {}
        outcomes: dict[AgentId, ActionOutcome] = {}

        for agent in active:
            raw = actions[agent]
            try:
                parsed[agent] = (
                    parse_action(raw) if isinstance(raw, str) else raw
                )
            except ActionParseError as error:
                parsed[agent] = None
                outcomes[agent] = self._finish(
                    agent, None, False, f"Invalid action: {error}"
                )

        take_winners: dict[str, AgentId] = {}

        for phase in VERB_PHASES:
            for agent in order:
                if agent not in active or agent in outcomes:
                    continue
                action = parsed[agent]
                assert action is not None
                if action.verb not in phase:
                    continue
                if action.verb == "take":
                    item_id = action.args[0]
                    holder = take_winners.get(item_id)
                    if holder is not None and holder != agent:
                        outcomes[agent] = self._finish(
                            agent,
                            action,
                            False,
                            TAKE_CONFLICT_MESSAGE,
                        )
                        continue
                    outcome = self._apply_parsed_action(agent, action)
                    if outcome.valid:
                        take_winners[item_id] = agent
                    outcomes[agent] = outcome
                else:
                    outcomes[agent] = self._apply_parsed_action(agent, action)

        for agent in active:
            if agent not in outcomes:
                action = parsed[agent]
                outcomes[agent] = self._finish(
                    agent, action, False, "Unhandled action."
                )

        action_lines = {
            agent: (
                actions[agent]
                if isinstance(actions[agent], str)
                else str(actions[agent])
            )
            for agent in active
        }
        self._record_joint_step(action_lines)
        return outcomes

    def _record_joint_step(self, action_lines: dict[AgentId, str]) -> None:
        if not action_lines:
            return
        self.last_joint_actions.update(action_lines)
        self.joint_action_history.append(dict(action_lines))
        overflow = len(self.joint_action_history) - self._joint_history_limit
        if overflow > 0:
            del self.joint_action_history[:overflow]

    def action_rejection_reason(
        self,
        agent: AgentId,
        action: Action,
        *,
        committed: Mapping[AgentId, Action] | None = None,
    ) -> str | None:
        """Return why an action would fail now (read-only); ``None`` if it may succeed."""
        if action.verb == "say" and not self.worker_broadcast:
            return "Worker broadcast is unavailable. Use report(message) to contact the orchestrator."
        if self.communication == "disabled" and action.verb in {"say", "write_note", "read_notes", "give", "give_to", "report"}:
            return "Communication and item transfers are disabled for this episode."
        if action.verb == "report" and not self.coordination:
            return "There is no orchestrator for this episode."
        if action.verb == "move":
            return self._movement_rejection(agent, action.args[0])
        if action.verb == "take":
            item_id = action.args[0]
            if item_id not in self.room.items:
                return (
                    f"Unknown item: {item_id}. "
                    "Clues and notes use inspect(node_id), not take(...)."
                )
            if committed:
                for other, other_action in committed.items():
                    if (
                        other != agent
                        and other_action.verb == "take"
                        and other_action.args[0] == item_id
                    ):
                        return TAKE_CONFLICT_MESSAGE
            if item_id in self.taken_items:
                return "That item is no longer there."
            if self._available_item_source(agent, item_id) is None:
                return "That item is not available."

        if action.verb in {"enter", "use"}:
            node_id = action.args[-1]
            node, error = self._lockable_node(agent, node_id)
            if error is not None:
                return error
            assert node is not None
            if action.verb == "use":
                item_id = action.args[0]
                if item_id not in self.agents[agent].bag:
                    return "You are not carrying that item."

        if action.verb == "inspect":
            _node, error = self._usable_node(agent, action.args[0])
            if error is not None:
                return error

        if action.verb == "give_to":
            return self._transfer_rejection(agent, action.args[0], action.args[1])
        if action.verb == "give":
            if len(self.agents) > 2:
                return "Use give_to(item_id, agent_id) to select a recipient."
            item_id = action.args[0]
            if item_id not in self.agents[agent].bag:
                return "You are not carrying that item."
            partner = self._partner_of(agent)
            if partner is not None and self.agents[agent].location != self.agents[partner].location:
                return "Item transfers require both agents to be in the same room."

        return None

    def _can_act(self, agent: AgentId) -> bool:
        state = self.agents[agent]
        return not state.escaped and state.turns_left > 0

    def _apply_parsed_action(
        self, agent: AgentId, action: Action
    ) -> ActionOutcome:
        if action.verb == "say" and not self.worker_broadcast:
            return self._finish(agent, action, False, "Worker broadcast is unavailable. Use report(message) to contact the orchestrator.")
        if self.communication == "disabled" and action.verb in {"say", "write_note", "read_notes", "give", "give_to", "report"}:
            return self._finish(agent, action, False, self.action_rejection_reason(agent, action))
        handler = getattr(self, f"_act_{action.verb}")
        valid, result = handler(agent, *action.args)
        return self._finish(agent, action, valid, result)

    def _finish(
        self,
        agent: AgentId,
        action: Action | None,
        valid: bool,
        result: str,
    ) -> ActionOutcome:
        self.agents[agent].last_result = result
        return ActionOutcome(action=action, valid=valid, result=result)

    def _agent_state(self, agent: AgentId) -> AgentState:
        try:
            return self.agents[agent]
        except KeyError as error:
            raise ValueError(f"Agent {agent!r} does not exist in this room") from error

    def _usable_node(
        self, agent: AgentId, node_id: str
    ) -> tuple[Node | Passage | None, str | None]:
        passage = next((p for p in self.room.passages if p.id == node_id), None)
        if passage is not None:
            if passage not in self.visible_passages(agent):
                return None, "That passage is not adjacent to your room."
            return passage, None
        try:
            node = self.room.node(node_id)
        except KeyError:
            return None, f"Unknown node: {node_id}."
        if not self.room.is_shared and node.wing != agent:
            return None, "That node is not in your wing."
        if not self.is_visible(node):
            return None, "That node is hidden."
        if self.room.locations and node.location != self.agents[agent].location:
            return None, "That node is not in your current room."
        return node, None

    def _act_inspect(self, agent: AgentId, node_id: str) -> tuple[bool, str]:
        node, error = self._usable_node(agent, node_id)
        if error is not None:
            return False, error
        assert node is not None
        if isinstance(node, Node) and node.kind == "clue":
            value = self.room.codes[node.code] if node.code is not None else node.text
            assert value is not None
            self.agents[agent].read_codes[node.id] = value
            self.inspected_nodes.add(node_id)
            return True, f"{node.desc} It reads {value}."
        if isinstance(node, Node) and node.kind == "search":
            self.opened_nodes.add(node.id)
            self.inspected_nodes.add(node.id)
            contents = [content for content in node.contents
                        if content not in self.taken_items]
            return True, f"You search {node.id}. Revealed: {', '.join(contents) or '(nothing remaining)'}."
        self.inspected_nodes.add(node_id)
        return True, node.text or node.desc

    def _act_take(self, agent: AgentId, item_id: str) -> tuple[bool, str]:
        if item_id not in self.room.items:
            return False, f"Unknown item: {item_id}."
        if item_id in self.taken_items:
            return False, "That item is no longer there."

        source = self._available_item_source(agent, item_id)
        if source is None:
            return False, "That item is not available in your wing."

        self.taken_items.add(item_id)
        self.agents[agent].bag.add(item_id)
        return True, f"You take the {self.room.items[item_id].name}."

    def _available_item_source(
        self, agent: AgentId, item_id: str
    ) -> Node | None:
        for node in self.visible_nodes(agent):
            if node.kind == "spot" and item_id in node.items:
                return node
            if (
                node.kind in {"container", "search"}
                and node.id in self.opened_nodes
                and item_id in node.contents
            ):
                return node
        return None

    def _act_enter(
        self, agent: AgentId, code: str, node_id: str
    ) -> tuple[bool, str]:
        node, error = self._lockable_node(agent, node_id)
        if error is not None:
            return False, error
        assert node is not None and node.requires is not None
        if node.requires.code is None or self.room.codes[node.requires.code] != code:
            return True, "wrong"
        return self._open(agent, node)

    def _act_use(
        self, agent: AgentId, item_id: str, node_id: str
    ) -> tuple[bool, str]:
        state = self.agents[agent]
        if item_id not in state.bag:
            return False, "You are not carrying that item."
        node, error = self._lockable_node(agent, node_id)
        if error is not None:
            return False, error
        assert node is not None and node.requires is not None
        if item_id not in node.requires.items:
            return True, "wrong"
        return self._open(agent, node)

    def _lockable_node(
        self, agent: AgentId, node_id: str
    ) -> tuple[Node | Passage | None, str | None]:
        node, error = self._usable_node(agent, node_id)
        if error is not None:
            return None, error
        assert node is not None
        if node.kind not in {"container", "door", "passage"}:
            return None, "That node has no lock."
        if isinstance(node, Passage) and node.requires is None:
            return None, "That passage is already open."
        if node.id in self.opened_nodes:
            return None, "That node is already open."
        return node, None

    def _open(self, agent: AgentId, node: Node | Passage) -> tuple[bool, str]:
        self.opened_nodes.add(node.id)
        if node.kind == "door":
            if self.room.is_shared and self.shared_escape == "together":
                for state in self.agents.values():
                    state.escaped = True
                return True, "The door opens. Everyone escapes!"
            self.agents[agent].escaped = True
            if self.room.is_shared and self.shared_escape == "race":
                return True, "The door opens. You escape! The other agent is still inside."
            return True, "The door opens. You escape!"
        return True, f"You open {node.id}."

    def _act_give(self, agent: AgentId, item_id: str) -> tuple[bool, str]:
        if len(self.agents) > 2:
            return False, "Use give_to(item_id, agent_id) to select a recipient."
        partner = self._partner_of(agent)
        if partner is None:
            return False, "There is no partner in this room."
        if item_id not in self.agents[agent].bag:
            return False, "You are not carrying that item."
        if self.agents[agent].location != self.agents[partner].location:
            return False, "Item transfers require both agents to be in the same room."
        self.agents[agent].bag.remove(item_id)
        self.agents[partner].bag.add(item_id)
        return True, f"You give the {self.room.items[item_id].name} to {partner}."

    def _transfer_rejection(self, agent: AgentId, item_id: str, recipient: AgentId) -> str | None:
        if recipient not in self.agents or recipient == agent:
            return "Choose a different agent in this team as the recipient."
        if item_id not in self.agents[agent].bag:
            return "You are not carrying that item."
        if self.agents[agent].location != self.agents[recipient].location:
            return "Item transfers require both agents to be in the same room."
        return None

    def _act_give_to(self, agent: AgentId, item_id: str, recipient: AgentId) -> tuple[bool, str]:
        error = self._transfer_rejection(agent, item_id, recipient)
        if error:
            return False, error
        self.agents[agent].bag.remove(item_id)
        self.agents[recipient].bag.add(item_id)
        return True, f"You give the {self.room.items[item_id].name} to {recipient}."

    def _act_report(self, agent: AgentId, text: str) -> tuple[bool, str]:
        if not self.coordination:
            return False, "There is no orchestrator for this episode."
        self.reports.append(Message(sender=agent, text=self._truncate_message(text)))
        return True, "reported to orchestrator"

    def _movement_rejection(self, agent: AgentId, room_id: str) -> str | None:
        current = self.agents[agent].location
        if room_id == current:
            return "You are already in that room."
        passage = next((p for p in self.visible_passages(agent) if room_id in p.locations), None)
        if passage is None:
            return "That room is not adjacent to your current room."
        if not self.passage_is_open(passage):
            return f"Passage {passage.id} is locked."
        return None

    def _act_move(self, agent: AgentId, room_id: str) -> tuple[bool, str]:
        error = self._movement_rejection(agent, room_id)
        if error:
            return False, error
        self.agents[agent].location = room_id
        return True, f"You move to {room_id} ({self.room.locations[room_id]})."

    def _act_say(self, agent: AgentId, text: str) -> tuple[bool, str]:
        if self._partner_of(agent) is None:
            return False, "There is no partner in this room."
        message = self._truncate_message(text)
        self.messages.append(Message(sender=agent, text=message))
        return True, "sent"

    def _act_write_note(self, agent: AgentId, text: str) -> tuple[bool, str]:
        note = self._truncate_message(text)
        self.notes.append(note)
        return True, "noted"

    def _act_read_notes(self, agent: AgentId) -> tuple[bool, str]:
        del agent
        return True, "\n".join(self.notes) if self.notes else "(no notes)"

    def _act_wait(self, agent: AgentId) -> tuple[bool, str]:
        del agent
        return True, "You wait."

    def _partner_of(self, agent: AgentId) -> AgentId | None:
        for candidate in self.agents:
            if candidate != agent:
                return candidate
        return None

    def _truncate_message(self, text: str) -> str:
        return " ".join(text.split()[: self.max_message_words])
