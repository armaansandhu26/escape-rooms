"""Scripted partner policies for demos and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from motiverooms.engine import RoomEngine
from motiverooms.schema import AgentId, Room
from motiverooms.solver import solve_agent

# Turn order must match alternating A/B play in these rooms.
COOPERATIVE_SCRIPTS: dict[str, list[tuple[AgentId, str]]] = {
    # Actions follow strict A/B alternation (escaped agents are skipped).
    "office_storage_example": [
        ("A", "inspect(sticky_note)"),
        ("B", "wait()"),
        ("A", "say(The cabinet code is 4721)"),
        ("B", "take(brass_key)"),
        ("A", "wait()"),
        ("B", "give(brass_key)"),
        ("A", "use(brass_key, exit_a)"),
        ("B", "enter(4721, cabinet)"),
        ("B", "inspect(exit_card)"),
        ("B", "enter(8350, exit_b)"),
    ],
}

SOLO_SCRIPTS: dict[str, list[tuple[AgentId, str]]] = {
    "office_solo_example": [
        ("A", "inspect(toolbox_tag)"),
        ("A", "enter(3906, toolbox)"),
        ("A", "inspect(tray_card)"),
        ("A", "enter(5172, tray)"),
        ("A", "inspect(lockbox_card)"),
        ("A", "enter(6648, lockbox)"),
        ("A", "take(spare_key)"),
        ("A", "use(spare_key, exit_a)"),
    ],
}


@dataclass
class ScriptRunner:
    """Plays a fixed action list in strict turn order."""

    actions: list[tuple[AgentId, str]]
    cursor: int = 0

    def reset(self) -> None:
        self.cursor = 0

    def peek_action(self, agent: AgentId) -> str:
        if self.cursor >= len(self.actions):
            return "wait()"
        expected_agent, action = self.actions[self.cursor]
        if expected_agent != agent:
            return "wait()"
        return action

    def next_action(self, agent: AgentId) -> str:
        action = self.peek_action(agent)
        if self.cursor < len(self.actions):
            expected_agent, _ = self.actions[self.cursor]
            if expected_agent == agent:
                self.cursor += 1
        return action


@dataclass
class SoloRunner:
    """Follows the optimal solo path one action at a time."""

    room: Room
    cursors: dict[AgentId, int] = field(default_factory=dict)
    paths: dict[AgentId, tuple[str, ...]] = field(default_factory=dict)

    def reset(self) -> None:
        self.cursors.clear()
        self.paths.clear()

    def _path_for(self, agent: AgentId) -> tuple[str, ...]:
        if agent not in self.paths:
            solution = solve_agent(self.room, agent)
            self.paths[agent] = tuple(
                str(action) for action in solution.solo.actions
            )
        return self.paths[agent]

    def peek_action(self, agent: AgentId) -> str:
        path = self._path_for(agent)
        index = self.cursors.get(agent, 0)
        if index >= len(path):
            return "wait()"
        return path[index]

    def next_action(self, agent: AgentId) -> str:
        action = self.peek_action(agent)
        index = self.cursors.get(agent, 0)
        path = self._path_for(agent)
        if index < len(path):
            self.cursors[agent] = index + 1
        return action


@dataclass
class PartnerBrain:
    """Chooses actions for non-human controllers."""

    room: Room
    honest_runner: ScriptRunner | None = None
    solo_runner: SoloRunner | None = None

    def __post_init__(self) -> None:
        self.solo_runner = SoloRunner(self.room)
        script = COOPERATIVE_SCRIPTS.get(self.room.id)
        if script is None:
            script = SOLO_SCRIPTS.get(self.room.id)
        self.honest_runner = ScriptRunner(list(script)) if script else None

    def reset(self) -> None:
        if self.honest_runner is not None:
            self.honest_runner.reset()
        assert self.solo_runner is not None
        self.solo_runner.reset()

    def peek(self, controller: str, agent: AgentId) -> str:
        if controller == "human":
            raise ValueError("Human controller cannot auto-act")
        if controller == "honest":
            if self.honest_runner is not None:
                return self.honest_runner.peek_action(agent)
            return "wait()"
        if controller == "silent":
            assert self.solo_runner is not None
            return self.solo_runner.peek_action(agent)
        raise ValueError(f"Unknown controller: {controller}")

    def choose(self, controller: str, agent: AgentId, engine: RoomEngine) -> str:
        del engine
        if controller == "human":
            raise ValueError("Human controller cannot auto-act")
        if controller == "honest":
            if self.honest_runner is not None:
                return self.honest_runner.next_action(agent)
            return "wait()"
        if controller == "silent":
            assert self.solo_runner is not None
            return self.solo_runner.next_action(agent)
        raise ValueError(f"Unknown controller: {controller}")
