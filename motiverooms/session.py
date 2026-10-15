"""Browser session: room, controllers, episode log, and scripted steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from motiverooms.catalog import RoomEntry, room_by_key
from motiverooms.engine import ActionOutcome
from motiverooms.narrative import narrate_turn
from motiverooms.partners import PartnerBrain
from motiverooms.schema import AgentId, load_room
from motiverooms.web_game import WebGame


@dataclass
class EpisodeTurnRecord:
    env_step: int
    agent: AgentId
    controller: str
    action: str
    valid: bool
    result: str
    tag: str
    narrative: str


@dataclass
class AppSession:
    room_entry: RoomEntry
    turn_budget: int = 12
    mode: str = "collaborative"
    controllers: dict[AgentId, str] = field(default_factory=dict)
    episode_log: list[EpisodeTurnRecord] = field(default_factory=list)
    game: WebGame = field(init=False)
    brain: PartnerBrain = field(init=False)

    def __post_init__(self) -> None:
        self.reload_game()

    def reload_game(self) -> None:
        room = load_room(self.room_entry.path)
        self.game = WebGame(room, turn_budget=self.turn_budget, mode=self.mode)
        self.brain = PartnerBrain(room, mode=self.mode)
        if not self.controllers:
            self.controllers = {
                agent: "human" for agent in room.wings  # type: ignore[misc]
            }

    def reset_episode(self) -> None:
        self.game.reset()
        self.brain.reset()
        self.episode_log.clear()

    def controller_for(self, agent: AgentId) -> str:
        return self.controllers.get(agent, "human")

    def is_scripted_turn(self) -> bool:
        """True when every active agent is non-human (one joint step can run)."""
        return self.is_scripted_step()

    def is_scripted_step(self) -> bool:
        if self.game.round_over:
            return False
        active = self.game.active_agents
        if not active:
            return False
        return all(self.controller_for(agent) != "human" for agent in active)

    def peek_joint_actions(self) -> dict[AgentId, str]:
        active = self.game.active_agents
        joint = self.brain.peek_joint_step()
        if joint is not None:
            return {agent: joint[agent] for agent in active if agent in joint}
        return {
            agent: self.brain.peek(self.controller_for(agent), agent)
            for agent in active
        }

    def peek_scripted_turn(self) -> dict[str, Any]:
        if not self.is_scripted_step():
            raise ValueError("Not all active agents are scripted")
        actions = self.peek_joint_actions()
        return {
            "env_step": self.game.env_step + 1,
            "actions": actions,
            "agents": list(actions),
        }

    def submit_human(self, action: str) -> ActionOutcome:
        agent = self.game.active_agent
        if agent is None:
            raise ValueError("The round is over")
        if self.controller_for(agent) != "human":
            raise ValueError(
                f"Agent {agent} is controlled by {self.controller_for(agent)}"
            )
        outcomes = self._submit_joint({agent: action})
        return outcomes[agent]

    def submit_auto(self) -> dict[AgentId, ActionOutcome]:
        if not self.is_scripted_step():
            raise ValueError("The active agent is human-controlled")
        active = self.game.active_agents
        joint = self.brain.next_joint_step()
        if joint is not None:
            actions = {agent: joint[agent] for agent in active if agent in joint}
        else:
            actions = {
                agent: self.brain.choose(
                    self.controller_for(agent), agent, self.game.engine
                )
                for agent in active
            }
        return self._submit_joint(actions)

    def _submit_joint(
        self, actions: dict[AgentId, str]
    ) -> dict[AgentId, ActionOutcome]:
        if self.game.round_over:
            raise ValueError("The round is over")
        outcomes = self.game.submit_joint(actions)
        env_step = self.game.env_step
        for agent, outcome in outcomes.items():
            tag, narrative = narrate_turn(
                self.game.room,
                agent,
                actions[agent],
                outcome,
                engine=self.game.engine,
            )
            self.episode_log.append(
                EpisodeTurnRecord(
                    env_step=env_step,
                    agent=agent,
                    controller=self.controller_for(agent),
                    action=actions[agent],
                    valid=outcome.valid,
                    result=outcome.result,
                    tag=tag,
                    narrative=narrative,
                )
            )
        return outcomes

    def run_scripted_episode(self, *, max_steps: int = 100) -> int:
        steps = 0
        while not self.game.round_over and steps < max_steps:
            if not self.is_scripted_step():
                break
            self.submit_auto()
            steps += 1
        return steps

    def configure(
        self,
        *,
        room_key: str,
        mode: str,
        partner_a: str,
        partner_b: str,
        turn_budget: int | None = None,
    ) -> None:
        self.room_entry = room_by_key(room_key)
        self.mode = mode
        if turn_budget is not None:
            self.turn_budget = turn_budget
        room = load_room(self.room_entry.path)
        self.controllers = {}
        if "A" in room.wings:
            self.controllers["A"] = partner_a
        if "B" in room.wings:
            self.controllers["B"] = partner_b
        self.reload_game()
        self.reset_episode()

    def simulation_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "controllers": dict(self.controllers),
            "room_key": self.room_entry.key,
        }

    def episode_log_payload(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.episode_log]

    def episode_steps_payload(self) -> list[dict[str, Any]]:
        grouped: dict[int, dict[str, Any]] = {}
        order: list[int] = []
        for record in self.episode_log:
            if record.env_step not in grouped:
                grouped[record.env_step] = {
                    "env_step": record.env_step,
                    "agents": {},
                }
                order.append(record.env_step)
            grouped[record.env_step]["agents"][record.agent] = {
                "controller": record.controller,
                "action": record.action,
                "valid": record.valid,
                "result": record.result,
                "tag": record.tag,
                "narrative": record.narrative,
            }
        return [grouped[step] for step in order]

    def state_payload(self) -> dict[str, Any]:
        payload = self.game.state_payload()
        payload["simulation"] = self.simulation_payload()
        payload["episode_log"] = self.episode_log_payload()
        payload["episode_steps"] = self.episode_steps_payload()
        for agent, controller in self.controllers.items():
            if agent in payload["agents"]:
                payload["agents"][agent]["controller"] = controller
        payload["scripted_turn"] = self.is_scripted_step()
        payload["scripted_step"] = self.is_scripted_step()
        return payload
