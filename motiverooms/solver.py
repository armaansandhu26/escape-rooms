"""Breadth-first oracle for solo and instantly-assisted escape paths."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from motiverooms.actions import Action
from motiverooms.schema import AgentId, Node, Room


@dataclass(frozen=True)
class SearchState:
    opened_nodes: frozenset[str] = frozenset()
    known_codes: frozenset[str] = frozenset()
    bag: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PathResult:
    turns: int | float
    actions: tuple[Action, ...]

    @property
    def reachable(self) -> bool:
        return not math.isinf(self.turns)


@dataclass(frozen=True)
class AgentSolution:
    solo: PathResult
    cooperative: PathResult

    @property
    def help_value(self) -> int | float:
        return self.solo.turns - self.cooperative.turns


def solve_agent(room: Room, agent: AgentId) -> AgentSolution:
    """Return optimal solo and free-help paths for one agent."""

    if agent not in room.wings:
        raise ValueError(f"Agent {agent!r} does not exist in this room")
    return AgentSolution(
        solo=shortest_path(room, agent, cooperative=False),
        cooperative=shortest_path(room, agent, cooperative=True),
    )


def solve_room(room: Room) -> dict[AgentId, AgentSolution]:
    return {agent: solve_agent(room, agent) for agent in room.wings}


def shortest_path(
    room: Room,
    agent: AgentId,
    *,
    cooperative: bool,
) -> PathResult:
    """Find the fewest own turns needed to open ``agent``'s exit.

    In cooperative mode, requirements available anywhere in the partner's
    wing are placed in the initial state for free.  This measures the value of
    instant help without including the partner's communication turns.
    """

    initial = SearchState()
    if cooperative:
        codes, items = _handoff_resources(room, agent)
        initial = SearchState(known_codes=codes, bag=items)

    queue: deque[tuple[SearchState, tuple[Action, ...]]] = deque([(initial, ())])
    visited = {initial}

    while queue:
        state, path = queue.popleft()
        for action, next_state, escaped in _transitions(room, agent, state):
            next_path = path + (action,)
            if escaped:
                return PathResult(turns=len(next_path), actions=next_path)
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, next_path))

    return PathResult(turns=math.inf, actions=())


def _transitions(
    room: Room,
    agent: AgentId,
    state: SearchState,
) -> list[tuple[Action, SearchState, bool]]:
    transitions: list[tuple[Action, SearchState, bool]] = []
    visible_nodes = [
        node
        for node in room.nodes_in_wing(agent)
        if _is_visible(room, node, state.opened_nodes)
    ]

    for node in visible_nodes:
        if node.kind == "clue" and node.code not in state.known_codes:
            assert node.code is not None
            next_state = SearchState(
                opened_nodes=state.opened_nodes,
                known_codes=state.known_codes | {node.code},
                bag=state.bag,
            )
            transitions.append((Action("inspect", (node.id,)), next_state, False))

        for item_id in _available_items(room, node, state):
            next_state = SearchState(
                opened_nodes=state.opened_nodes,
                known_codes=state.known_codes,
                bag=state.bag | {item_id},
            )
            transitions.append((Action("take", (item_id,)), next_state, False))

        if node.kind not in {"container", "door"}:
            continue
        if node.id in state.opened_nodes:
            continue
        assert node.requires is not None

        action: Action | None = None
        if (
            node.requires.code is not None
            and node.requires.code in state.known_codes
        ):
            value = room.codes[node.requires.code]
            action = Action("enter", (value, node.id))
        else:
            usable_item = next(
                (item for item in node.requires.items if item in state.bag),
                None,
            )
            if usable_item is not None:
                action = Action("use", (usable_item, node.id))

        if action is not None:
            next_state = SearchState(
                opened_nodes=state.opened_nodes | {node.id},
                known_codes=state.known_codes,
                bag=state.bag,
            )
            transitions.append((action, next_state, node.kind == "door"))

    return transitions


def _is_visible(room: Room, node: Node, opened_nodes: frozenset[str]) -> bool:
    if not node.hidden:
        return True
    return any(
        parent.id in opened_nodes and node.id in parent.contents
        for parent in room.nodes
        if parent.kind == "container"
    )


def _available_items(
    room: Room, node: Node, state: SearchState
) -> tuple[str, ...]:
    if node.kind == "spot":
        candidates = node.items
    elif node.kind == "container" and node.id in state.opened_nodes:
        candidates = node.contents
    else:
        candidates = ()
    return tuple(
        item
        for item in candidates
        if item in room.items and item not in state.bag
    )


def _handoff_resources(
    room: Room, agent: AgentId
) -> tuple[frozenset[str], frozenset[str]]:
    partner_wings = set(room.wings) - {agent}
    if not partner_wings:
        return frozenset(), frozenset()

    needed_codes: set[str] = set()
    needed_items: set[str] = set()
    for node in room.nodes_in_wing(agent):
        if node.requires is None:
            continue
        if node.requires.code is not None:
            needed_codes.add(node.requires.code)
        needed_items.update(node.requires.items)

    partner_codes = {
        node.code
        for node in room.nodes
        if node.wing in partner_wings and node.kind == "clue" and node.code is not None
    }
    partner_items: set[str] = set()
    for node in room.nodes:
        if node.wing not in partner_wings:
            continue
        partner_items.update(node.items)
        partner_items.update(
            content for content in node.contents if content in room.items
        )

    return (
        frozenset(needed_codes & partner_codes),
        frozenset(needed_items & partner_items),
    )
