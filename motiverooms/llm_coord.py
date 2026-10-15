"""LLM rollout coordination (staggered picks within one joint step)."""

from __future__ import annotations

from collections.abc import Callable

from motiverooms.actions import Action, parse_action
from motiverooms.engine import RoomEngine
from motiverooms.llm_policy import LLMAgentPolicy
from motiverooms.schema import AgentId

DEFAULT_MAX_ACTION_ATTEMPTS = 4


def decision_order(active: tuple[AgentId, ...], env_step: int) -> tuple[AgentId, ...]:
    """Alternate who chooses first so neither seat always wins ties."""
    ordered = tuple(sorted(active))
    if env_step % 2 == 1:
        return tuple(reversed(ordered))
    return ordered


def with_partner_pick(observation: str, partner: AgentId, partner_action: str) -> str:
    return (
        f"{observation}\n\n"
        "PARTNER PICK THIS STEP (same simultaneous tick — choose a different action):\n"
        f"- Agent {partner}: {partner_action}"
    )


def with_rejection_feedback(observation: str, reason: str) -> str:
    return (
        f"{observation}\n\n"
        f"That choice will not work: {reason}\n"
        "Pick a different action on the last line."
    )


def choose_action_with_checks(
    policy: LLMAgentPolicy,
    observation: str,
    engine: RoomEngine,
    agent: AgentId,
    committed: dict[AgentId, Action],
    *,
    max_attempts: int = DEFAULT_MAX_ACTION_ATTEMPTS,
    on_llm_wait: Callable[[AgentId, int], None] | None = None,
) -> str:
    """Ask the LLM for an action; re-prompt when the engine would reject it."""
    obs = observation
    action_line = ""
    for attempt in range(max_attempts):
        if on_llm_wait is not None:
            on_llm_wait(agent, attempt + 1)
        _, action_line = policy.choose_action(obs)
        reason = engine.action_rejection_reason(
            agent,
            parse_action(action_line),
            committed=committed,
        )
        if reason is None:
            return action_line
        obs = with_rejection_feedback(obs, reason)
    return action_line
