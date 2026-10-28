"""Gym-style wrapper around the joint-step room engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from motiverooms.engine import ActionOutcome, RoomEngine, SharedEscapeMode
from motiverooms.render import render_observation
from motiverooms.rewards import MODES, terminal_rewards
from motiverooms.schema import AgentId, Room, load_room


@dataclass
class MotiveRoomsEnv:
    """One multi-agent room episode with simultaneous ``step_joint`` ticks."""

    room: Room
    turn_budget: int = 14
    framing: str = "collaborative"  # Compatibility metadata; prompts belong to policies.
    round_number: int = 1
    total_rounds: int = 1
    mode: str = "collaborative"
    shared_escape: SharedEscapeMode = "together"
    partner_actions: str = "private"
    observation_version: str = "v3"
    communication: str = "enabled"
    coordination: bool = False
    worker_broadcast: bool = True
    engine: RoomEngine = field(init=False)
    env_step: int = field(init=False, default=0)

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs: object) -> MotiveRoomsEnv:
        return cls(room=load_room(path), **kwargs)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"Unknown reward mode: {self.mode}")
        if self.shared_escape not in {"together", "race"}:
            raise ValueError(f"Unknown shared escape rule: {self.shared_escape}")
        if self.partner_actions not in {"private", "public"}:
            raise ValueError(f"Unknown partner action visibility: {self.partner_actions}")
        if self.communication not in {"enabled", "disabled"}:
            raise ValueError(f"Unknown communication mode: {self.communication}")
        if self.communication == "disabled" and self.partner_actions != "private":
            raise ValueError("Disabled communication requires private partner actions")
        if self.coordination and self.communication == "disabled":
            raise ValueError("Orchestration requires enabled communication; use a fixed team for communication-off controls")
        if self.observation_version not in {"v1", "v2", "v3"}:
            raise ValueError(f"Unknown observation version: {self.observation_version}")
        self.reset()

    def reset(self) -> dict[AgentId, str]:
        self.engine = RoomEngine(
            self.room,
            turn_budget=self.turn_budget,
            shared_escape=self.shared_escape,
            communication=self.communication,
            coordination=self.coordination,
            worker_broadcast=self.worker_broadcast,
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
            self.room.is_shared
            and self.engine.shared_escape == "race"
            and any(state.escaped for state in self.engine.agents.values())
        ):
            return True
        return not self.active_agents()

    def observe(self, agent: AgentId) -> str:
        observation = render_observation(
            self.engine,
            agent,
            round_number=self.round_number,
            total_rounds=self.total_rounds,
            partner_actions=self.partner_actions,
            observation_version=self.observation_version,
        )
        if self.communication == "disabled":
            observation = observation.replace(
                "Clue readings and partner actions are private; messages and shared notes can convey codes.",
                "Clue readings and partner actions are private. Communication and item transfers are disabled.",
            )
            observation += "\nCommunication is disabled: say, write_note, read_notes and give are unavailable."
        if not self.worker_broadcast:
            observation = observation.replace(
                "Clue readings and partner actions are private; messages and shared notes can convey codes.",
                "Clue readings and partner actions are private. Worker broadcast is unavailable.",
            )
            observation += "\nWorker broadcast is unavailable. Use report(message) to contact the orchestrator."
        return observation

    def observe_all(self) -> dict[AgentId, str]:
        return {agent: self.observe(agent) for agent in self.active_agents()}

    def step_joint(self, actions: dict[AgentId, str]) -> dict[AgentId, ActionOutcome]:
        if self.round_over:
            raise ValueError("The round is over")
        outcomes = self.engine.step_joint(actions)
        self.env_step += 1
        return outcomes

    def rewards(self) -> dict[AgentId, float]:
        """Terminal scores only; framing never changes this transform."""
        if not self.round_over:
            raise ValueError("Rewards are available only at round end")
        return terminal_rewards(self.engine, self.mode)
