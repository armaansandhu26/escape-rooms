"""Browser session: room, controllers, episode log, and scripted steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from motiverooms.catalog import RoomEntry, room_by_key
from motiverooms.engine import ActionOutcome
from motiverooms.partners import PartnerBrain
from motiverooms.schema import AgentId, load_room
from motiverooms.web_game import WebGame


@dataclass
class EpisodeTurnRecord:
    turn: int
    agent: AgentId
    controller: str
    action: str
    valid: bool
    result: str


@dataclass
class AppSession:
    room_entry: RoomEntry
    turn_budget: int = 12
    mode: str = "collaborative"
    controllers: dict[AgentId, str] = field(default_factory=dict)
    episode_log: list[EpisodeTurnRecord] = field(default_factory=list)
    game: WebGame = field(init=False)
    brain: PartnerBrain = field(init=False)
    _turn_counter: int = 0

    def __post_init__(self) -> None:
        self.reload_game()

    def reload_game(self) -> None:
        room = load_room(self.room_entry.path)
        self.game = WebGame(room, turn_budget=self.turn_budget)
        self.brain = PartnerBrain(room)
        if not self.controllers:
            self.controllers = {
                agent: "human" for agent in room.wings  # type: ignore[misc]
            }

    def reset_episode(self) -> None:
        self.game.reset()
        self.brain.reset()
        self.episode_log.clear()
        self._turn_counter = 0

    def controller_for(self, agent: AgentId) -> str:
        return self.controllers.get(agent, "human")

    def is_scripted_turn(self) -> bool:
        agent = self.game.active_agent
        return agent is not None and self.controller_for(agent) != "human"

    def choose_scripted_action(self) -> str:
        agent = self.game.active_agent
        if agent is None:
            raise ValueError("No active agent")
        return self.brain.choose(
            self.controller_for(agent), agent, self.game.engine
        )

    def peek_scripted_turn(self) -> dict[str, str]:
        agent = self.game.active_agent
        if agent is None:
            raise ValueError("No active agent")
        if not self.is_scripted_turn():
            raise ValueError("The active agent is human-controlled")
        controller = self.controller_for(agent)
        return {
            "agent": agent,
            "controller": controller,
            "action": self.brain.peek(controller, agent),
        }

    def submit_human(self, action: str) -> ActionOutcome:
        agent = self.game.active_agent
        if agent is None:
            raise ValueError("The round is over")
        if self.controller_for(agent) != "human":
            raise ValueError(
                f"Agent {agent} is controlled by {self.controller_for(agent)}"
            )
        return self._submit(action, controller="human")

    def submit_auto(self) -> ActionOutcome:
        if not self.is_scripted_turn():
            raise ValueError("The active agent is human-controlled")
        action = self.choose_scripted_action()
        agent = self.game.active_agent
        assert agent is not None
        return self._submit(action, controller=self.controller_for(agent))

    def _submit(self, action: str, *, controller: str) -> ActionOutcome:
        agent = self.game.active_agent
        if agent is None:
            raise ValueError("The round is over")
        outcome = self.game.submit(action)
        self._turn_counter += 1
        self.episode_log.append(
            EpisodeTurnRecord(
                turn=self._turn_counter,
                agent=agent,
                controller=controller,
                action=action,
                valid=outcome.valid,
                result=outcome.result,
            )
        )
        return outcome

    def run_scripted_episode(self, *, max_steps: int = 100) -> int:
        steps = 0
        while not self.game.round_over and steps < max_steps:
            if not self.is_scripted_turn():
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

    def state_payload(self) -> dict[str, Any]:
        payload = self.game.state_payload()
        payload["simulation"] = self.simulation_payload()
        payload["episode_log"] = self.episode_log_payload()
        for agent, controller in self.controllers.items():
            if agent in payload["agents"]:
                payload["agents"][agent]["controller"] = controller
        payload["scripted_turn"] = self.is_scripted_turn()
        return payload
