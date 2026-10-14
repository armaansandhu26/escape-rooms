"""Deterministic state transitions for one MotiveRooms room."""

from __future__ import annotations

from dataclasses import dataclass, field

from collections.abc import Mapping

from motiverooms.actions import Action, ActionParseError, parse_action
from motiverooms.joint_step import TAKE_CONFLICT_MESSAGE, VERB_PHASES
from motiverooms.schema import AgentId, Node, Room


@dataclass
class AgentState:
    turns_left: int
    bag: set[str] = field(default_factory=set)
    read_codes: dict[str, str] = field(default_factory=dict)
    escaped: bool = False
    last_result: str = "The round has not started."


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
    ) -> None:
        if turn_budget <= 0:
            raise ValueError("turn_budget must be positive")
        self.room = room
        self.max_message_words = max_message_words
        self.agents = {
            agent: AgentState(turns_left=turn_budget) for agent in room.wings
        }
        self.opened_nodes: set[str] = set()
        self.taken_items: set[str] = set()
        self.messages: list[Message] = []
        self.notes: list[str] = []
        self.inspected_nodes: set[str] = set()

    def is_visible(self, node: Node) -> bool:
        if not node.hidden:
            return True
        return any(
            parent.id in self.opened_nodes and node.id in parent.contents
            for parent in self.room.nodes
            if parent.kind == "container"
        )

    def visible_nodes(self, agent: AgentId) -> tuple[Node, ...]:
        return tuple(
            node
            for node in self.room.nodes_for_agent(agent)
            if self.is_visible(node)
        )

    def step(self, agent: AgentId, output: str | Action) -> ActionOutcome:
        state = self._agent_state(agent)
        if state.escaped:
            return ActionOutcome(None, False, "You have already escaped.")
        if state.turns_left <= 0:
            return ActionOutcome(None, False, "You have no turns left.")

        state.turns_left -= 1
        try:
            action = parse_action(output) if isinstance(output, str) else output
        except ActionParseError as error:
            return self._finish(agent, None, False, f"Invalid action: {error}")

        return self._apply_parsed_action(agent, action)

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

        return outcomes

    def _can_act(self, agent: AgentId) -> bool:
        state = self.agents[agent]
        return not state.escaped and state.turns_left > 0

    def _apply_parsed_action(
        self, agent: AgentId, action: Action
    ) -> ActionOutcome:
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
    ) -> tuple[Node | None, str | None]:
        try:
            node = self.room.node(node_id)
        except KeyError:
            return None, f"Unknown node: {node_id}."
        if not self.room.is_shared and node.wing != agent:
            return None, "That node is not in your wing."
        if not self.is_visible(node):
            return None, "That node is hidden."
        return node, None

    def _act_inspect(self, agent: AgentId, node_id: str) -> tuple[bool, str]:
        node, error = self._usable_node(agent, node_id)
        if error is not None:
            return False, error
        assert node is not None
        if node.kind == "clue":
            assert node.code is not None
            value = self.room.codes[node.code]
            self.agents[agent].read_codes[node.id] = value
            self.inspected_nodes.add(node_id)
            return True, f"{node.desc} It reads {value}."
        self.inspected_nodes.add(node_id)
        return True, node.desc

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
                node.kind == "container"
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
    ) -> tuple[Node | None, str | None]:
        node, error = self._usable_node(agent, node_id)
        if error is not None:
            return None, error
        assert node is not None
        if node.kind not in {"container", "door"}:
            return None, "That node has no lock."
        if node.id in self.opened_nodes:
            return None, "That node is already open."
        return node, None

    def _open(self, agent: AgentId, node: Node) -> tuple[bool, str]:
        self.opened_nodes.add(node.id)
        if node.kind == "door":
            if self.room.is_shared:
                for state in self.agents.values():
                    state.escaped = True
                return True, "The door opens. Everyone escapes!"
            self.agents[agent].escaped = True
            return True, "The door opens. You escape!"
        return True, f"You open {node.id}."

    def _act_give(self, agent: AgentId, item_id: str) -> tuple[bool, str]:
        partner = self._partner_of(agent)
        if partner is None:
            return False, "There is no partner in this room."
        if item_id not in self.agents[agent].bag:
            return False, "You are not carrying that item."
        self.agents[agent].bag.remove(item_id)
        self.agents[partner].bag.add(item_id)
        return True, f"You give the {self.room.items[item_id].name} to {partner}."

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
