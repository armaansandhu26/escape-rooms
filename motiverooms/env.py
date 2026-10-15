"""Gym-style wrapper around the joint-step room engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from motiverooms.engine import ActionOutcome, RoomEngine
from motiverooms.render import render_observation
from motiverooms.schema import AgentId, Room, load_room


@dataclass
class MotiveRoomsEnv:
    """One multi-agent room episode with simultaneous ``step_joint`` ticks."""

    room: Room
    turn_budget: int = 14
    framing: str = "collaborative"
    round_number: int = 1
    total_rounds: int = 1
    engine: RoomEngine = field(init=False)
    env_step: int = field(init=False, default=0)

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs: object) -> MotiveRoomsEnv:
        return cls(room=load_room(path), **kwargs)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> dict[AgentId, str]:
        shared_escape = (
            "race"
            if self.room.is_shared and self.framing == "competitive"
            else "together"
        )
        self.engine = RoomEngine(
            self.room,
            turn_budget=self.turn_budget,
            shared_escape=shared_escape,
        )
        self.env_step = 0
        return self.observe_all()

    @property
    def agent_ids(self) -> tuple[AgentId, ...]:
        return tuple(self.room.wings)

    def active_agents(self) -> tuple[AgentId, ...]:
        return tuple(
            agent
            for agent in self.agent_ids
            if not self.engine.agents[agent].escaped
            and self.engine.agents[agent].turns_left > 0
        )

    @property
    def round_over(self) -> bool:
        if (
            self.engine.shared_escape == "race"
            and any(state.escaped for state in self.engine.agents.values())
        ):
            return True
        return not self.active_agents()

    def observe(self, agent: AgentId) -> str:
        return render_observation(
            self.engine,
            agent,
            round_number=self.round_number,
            total_rounds=self.total_rounds,
        )

    def observe_all(self) -> dict[AgentId, str]:
        return {agent: self.observe(agent) for agent in self.active_agents()}

    def step_joint(self, actions: dict[AgentId, str]) -> dict[AgentId, ActionOutcome]:
        outcomes = self.engine.step_joint(actions)
        self.env_step += 1
        return outcomes
