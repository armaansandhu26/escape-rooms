"""Environment-step controller and observation payloads for the browser UI."""

from __future__ import annotations

from typing import Any

from motiverooms.engine import ActionOutcome, RoomEngine, SharedEscapeMode
from motiverooms.schema import AgentId, Node, Room


class WebGame:
    """One environment tick = all active agents act together via ``step_joint``."""

    def __init__(
        self,
        room: Room,
        turn_budget: int = 12,
        *,
        mode: str = "collaborative",
    ) -> None:
        self.room = room
        self.turn_budget = turn_budget
        self.mode = mode
        self.agent_order = tuple(room.wings)
        self.shared_escape: SharedEscapeMode = (
            "race" if room.is_shared and mode == "competitive" else "together"
        )
        self.engine = RoomEngine(
            room,
            turn_budget=turn_budget,
            shared_escape=self.shared_escape,
        )
        self.env_step: int = 0

    @property
    def active_agents(self) -> tuple[AgentId, ...]:
        return tuple(agent for agent in self.agent_order if self._can_act(agent))

    @property
    def active_agent(self) -> AgentId | None:
        """First active agent (UI highlight fallback)."""
        agents = self.active_agents
        return agents[0] if agents else None

    @property
    def round_over(self) -> bool:
        if self.shared_escape == "race" and any(
            state.escaped for state in self.engine.agents.values()
        ):
            return True
        return not self.active_agents

    @property
    def race_winner(self) -> AgentId | None:
        if self.shared_escape != "race":
            return None
        escaped = [a for a in self.agent_order if self.engine.agents[a].escaped]
        return escaped[0] if len(escaped) == 1 else None

    def reset(self) -> None:
        self.engine = RoomEngine(
            self.room,
            self.turn_budget,
            shared_escape=self.shared_escape,
        )
        self.env_step = 0

    def submit_joint(
        self, actions: dict[AgentId, str]
    ) -> dict[AgentId, ActionOutcome]:
        if self.round_over:
            raise ValueError("The round is over; reset to play again")
        outcomes = self.engine.step_joint(actions)
        self.env_step += 1
        return outcomes

    def submit(self, action_text: str) -> ActionOutcome:
        """Single-action submit for solo rooms or tests."""
        agent = self.active_agent
        if agent is None:
            raise ValueError("The round is over; reset to play again")
        return self.submit_joint({agent: action_text})[agent]

    @property
    def single_agent(self) -> bool:
        return len(self.room.wings) == 1

    def _can_act(self, agent: AgentId) -> bool:
        state = self.engine.agents[agent]
        return not state.escaped and state.turns_left > 0

    def state_payload(self) -> dict[str, Any]:
        active = self.active_agents
        wings = {
            agent: self._agent_payload(agent) for agent in self.room.wings
        }
        room_name = (
            self.room.space
            if self.room.is_shared
            else " × ".join(self.room.wings.values())
        )
        lead = active[0] if active else None
        return {
            "room": {
                "id": self.room.id,
                "name": room_name,
                "layout": self.room.layout,
                "space": self.room.space,
                "single_agent": self.single_agent,
            },
            "env_step": self.env_step,
            "active_agents": list(active),
            "active_agent": lead,
            "round_over": self.round_over,
            "race_winner": self.race_winner,
            "shared_escape": self.shared_escape,
            "agents": {
                agent: {
                    "wing": self._agent_space_label(agent),
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
            "view": wings[lead] if lead is not None else None,
            "messages": [
                {"sender": message.sender, "text": message.text}
                for message in self.engine.messages[-12:]
            ],
        }

    def _agent_payload(self, agent: AgentId) -> dict[str, Any]:
        state = self.engine.agents[agent]
        return {
            "agent": agent,
            "wing": self._agent_space_label(agent),
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
        if not self.room.is_shared and node.wing != agent:
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
            for candidate in self.room.nodes_for_agent(agent)
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

    def _agent_space_label(self, agent: AgentId) -> str:
        if self.room.is_shared:
            return self.room.space or self.room.wings[agent]
        return self.room.wings[agent]

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
