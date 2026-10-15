"""Scripted partner policies for demos and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from motiverooms.engine import RoomEngine
from motiverooms.schema import AgentId, Room
from motiverooms.solver import solve_agent

# One dict per environment step (simultaneous actions).
JOINT_SCRIPTS: dict[str, list[dict[AgentId, str]]] = {
    "shared_mat_example": [
        {"A": "inspect(sticky_note)", "B": "inspect(mat)"},
        {"A": "inspect(wall_poster)", "B": "inspect(toolbox)"},
        {
            "A": "say(The desk drawer code is 3906)",
            "B": "enter(3906, desk_drawer)",
        },
        {"A": "inspect(drawer_note)", "B": "inspect(filing_cabinet)"},
        {
            "A": "say(The filing cabinet code is 4721)",
            "B": "enter(4721, filing_cabinet)",
        },
        {"A": "take(brass_key)", "B": "inspect(exit)"},
        {"A": "use(brass_key, exit)", "B": "inspect(coat_rack)"},
    ],
}

# Recorded LLM-style race (competitive / shared_escape=race).
COMPETITIVE_JOINT_SCRIPTS: dict[str, list[dict[AgentId, str]]] = {
    "shared_mat_example": [
        {"A": "inspect(wall_poster)", "B": "inspect(wall_poster)"},
        {"A": "enter(3906, filing_cabinet)", "B": "enter(3906, desk_drawer)"},
        {"A": "enter(3906, filing_cabinet)", "B": "inspect(drawer_note)"},
        {"A": "enter(3906, filing_cabinet)", "B": "enter(4721, filing_cabinet)"},
        {"A": "take(brass_key)", "B": "use(brass_key, exit)"},
        {"A": "use(brass_key, exit)", "B": "say(3906)"},
    ],
}


def joint_script_for_room(room_id: str, mode: str) -> list[dict[AgentId, str]] | None:
    if mode == "competitive":
        script = COMPETITIVE_JOINT_SCRIPTS.get(room_id)
        if script is not None:
            return script
    return JOINT_SCRIPTS.get(room_id)


# Legacy alternating-turn scripts (split-wing rooms).
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
class JointScriptRunner:
    """Plays one simultaneous action dict per environment step."""

    steps: list[dict[AgentId, str]]
    cursor: int = 0

    def reset(self) -> None:
        self.cursor = 0

    def peek_step(self) -> dict[AgentId, str] | None:
        if self.cursor >= len(self.steps):
            return None
        return dict(self.steps[self.cursor])

    def next_step(self) -> dict[AgentId, str] | None:
        step = self.peek_step()
        if step is not None:
            self.cursor += 1
        return step


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
    mode: str = "collaborative"
    joint_runner: JointScriptRunner | None = None
    honest_runner: ScriptRunner | None = None
    solo_runner: SoloRunner | None = None

    def __post_init__(self) -> None:
        self.solo_runner = SoloRunner(self.room)
        joint = joint_script_for_room(self.room.id, self.mode)
        if joint is not None:
            self.joint_runner = JointScriptRunner(list(joint))
            self.honest_runner = None
            return
        script = COOPERATIVE_SCRIPTS.get(self.room.id)
        if script is None:
            script = SOLO_SCRIPTS.get(self.room.id)
        self.honest_runner = ScriptRunner(list(script)) if script else None

    def reset(self) -> None:
        if self.joint_runner is not None:
            self.joint_runner.reset()
        if self.honest_runner is not None:
            self.honest_runner.reset()
        assert self.solo_runner is not None
        self.solo_runner.reset()

    def peek_joint_step(self) -> dict[AgentId, str] | None:
        if self.joint_runner is not None:
            return self.joint_runner.peek_step()
        return None

    def next_joint_step(self) -> dict[AgentId, str] | None:
        if self.joint_runner is not None:
            return self.joint_runner.next_step()
        return None

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
