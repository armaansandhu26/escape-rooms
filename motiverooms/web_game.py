"""Turn controller and observation payloads for the browser UI."""

from __future__ import annotations

from typing import Any

from motiverooms.engine import ActionOutcome, RoomEngine
from motiverooms.schema import AgentId, Node, Room


class WebGame:
    """Small turn controller around the environment's room engine."""

    def __init__(self, room: Room, turn_budget: int = 12) -> None:
        self.room = room
        self.turn_budget = turn_budget
        self.agent_order = tuple(room.wings)
        self.engine = RoomEngine(room, turn_budget=turn_budget)
        self._current_index: int | None = 0

    @property
    def active_agent(self) -> AgentId | None:
        if self._current_index is None:
            return None
        return self.agent_order[self._current_index]

    @property
    def round_over(self) -> bool:
        return self.active_agent is None

    def reset(self) -> None:
        self.engine = RoomEngine(self.room, self.turn_budget)
        self._current_index = 0

    def submit(self, action_text: str) -> ActionOutcome:
        agent = self.active_agent
        if agent is None:
            raise ValueError("The round is over; reset to play again")
        outcome = self.engine.step(agent, action_text)
        self._advance_turn()
        return outcome

    @property
    def single_agent(self) -> bool:
        return len(self.room.wings) == 1

    def state_payload(self) -> dict[str, Any]:
        active_agent = self.active_agent
        wings = {
            agent: self._agent_payload(agent) for agent in self.room.wings
        }
        return {
            "room": {
                "id": self.room.id,
                "name": " × ".join(self.room.wings.values()),
                "single_agent": self.single_agent,
            },
            "active_agent": active_agent,
            "round_over": self.round_over,
            "agents": {
                agent: {
                    "wing": self.room.wings[agent],
                    "turns_left": state.turns_left,
                    "escaped": state.escaped,
                    "bag": [
                        {
                            "id": item_id,
                            "name": self.room.items[item_id].name,
                        }
                        for item_id in sorted(state.bag)
                    ],
                    "last_result": state.last_result,
                }
                for agent, state in self.engine.agents.items()
            },
            "wings": wings,
            "view": wings[active_agent] if active_agent is not None else None,
            "messages": [
                {"sender": message.sender, "text": message.text}
                for message in self.engine.messages[-12:]
            ],
        }

    def _advance_turn(self) -> None:
        assert self._current_index is not None
        for offset in range(1, len(self.agent_order) + 1):
            index = (self._current_index + offset) % len(self.agent_order)
            state = self.engine.agents[self.agent_order[index]]
            if not state.escaped and state.turns_left > 0:
                self._current_index = index
                return
        self._current_index = None

    def _agent_payload(self, agent: AgentId) -> dict[str, Any]:
        state = self.engine.agents[agent]
        return {
            "agent": agent,
            "wing": self.room.wings[agent],
            "turns_left": state.turns_left,
            "last_result": state.last_result,
            "nodes": [
                self._node_payload(agent, node)
                for node in self.engine.visible_nodes(agent)
            ],
        }

    def _node_payload(self, agent: AgentId, node: Node) -> dict[str, Any]:
        state = self.engine.agents[agent]
        payload: dict[str, Any] = {
            "id": node.id,
            "kind": node.kind,
            "description": node.desc,
            "status": self._node_status(agent, node),
            "opened": node.id in self.engine.opened_nodes,
            "position": self._node_position(agent, node),
        }
        payload["available_items"] = self._available_items_payload(agent, node)
        if node.kind in {"container", "door"} and node.requires is not None:
            payload["lock"] = (
                "code" if node.requires.code is not None else "item"
            )
            payload["usable_items"] = [
                {
                    "id": item_id,
                    "name": self.room.items[item_id].name,
                }
                for item_id in node.requires.items
                if item_id in state.bag
            ]
        return payload

    def _available_items_payload(
        self, agent: AgentId, node: Node
    ) -> list[dict[str, str]]:
        if node.wing != agent:
            return []
        if node.kind == "spot":
            candidates = node.items
        elif node.kind == "container" and node.id in self.engine.opened_nodes:
            candidates = node.contents
        else:
            return []
        return [
            {"id": item_id, "name": self.room.items[item_id].name}
            for item_id in candidates
            if item_id in self.room.items and item_id not in self.engine.taken_items
        ]

    def _node_position(self, agent: AgentId, node: Node) -> dict[str, float]:
        if node.kind == "door":
            return {"x": 94.0, "y": 50.0}

        objects = [
            candidate
            for candidate in self.room.nodes_in_wing(agent)
            if candidate.kind != "door"
        ]
        index = next(
            position
            for position, candidate in enumerate(objects)
            if candidate.id == node.id
        )
        columns = min(3, len(objects))
        rows = (len(objects) + columns - 1) // columns
        column = index % columns
        row = index // columns
        x = 20.0 if columns == 1 else 18.0 + column * (64.0 / (columns - 1))
        y = 50.0 if rows == 1 else 17.0 + row * (66.0 / (rows - 1))
        return {"x": round(x, 2), "y": round(y, 2)}

    def _node_status(self, agent: AgentId, node: Node) -> str:
        state = self.engine.agents[agent]
        if node.kind == "clue":
            code = state.read_codes.get(node.id)
            return f"Code {code}" if code is not None else "Unread clue"
        if node.kind == "spot":
            available = [
                self.room.items[item_id].name
                for item_id in node.items
                if item_id not in self.engine.taken_items
            ]
            return ", ".join(available) if available else "Empty"
        if node.id in self.engine.opened_nodes:
            if node.kind == "container":
                inside = self._available_items_payload(agent, node)
                if inside:
                    names = ", ".join(item["name"] for item in inside)
                    return f"Open · inside: {names}"
            return "Open"
        assert node.requires is not None
        return (
            "Needs a 4-digit code"
            if node.requires.code is not None
            else "Needs an item"
        )
