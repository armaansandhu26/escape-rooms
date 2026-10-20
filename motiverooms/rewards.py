"""Round payoffs and incentive transforms from the research specification."""

from motiverooms.engine import RoomEngine
from motiverooms.schema import AgentId

MODES = ("collaborative", "mixed", "competitive")


def terminal_rewards(engine: RoomEngine, mode: str) -> dict[AgentId, float]:
    if mode not in MODES:
        raise ValueError(f"Unknown reward mode: {mode}")
    payoffs = {
        agent: 10.0 + state.turns_left if state.escaped else 0.0
        for agent, state in engine.agents.items()
    }
    total = sum(payoffs.values())
    if mode == "collaborative":
        return {agent: total for agent in payoffs}
    if mode == "mixed":
        return payoffs
    return {agent: payoff - (total - payoff) for agent, payoff in payoffs.items()}
