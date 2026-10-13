#!/usr/bin/env python3
"""Play a room from the terminal as agent A, B, or both."""

from __future__ import annotations

import argparse
from pathlib import Path

from motiverooms.engine import RoomEngine
from motiverooms.render import render_observation
from motiverooms.schema import load_room

DEFAULT_ROOM = Path(__file__).parents[1] / "rooms/examples/office_storage.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("room", nargs="?", type=Path, default=DEFAULT_ROOM)
    parser.add_argument(
        "--agent",
        choices=("A", "B", "both"),
        default="both",
        help="Seat(s) controlled by the player; the other seat waits.",
    )
    parser.add_argument("--turn-budget", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    room = load_room(args.room)
    engine = RoomEngine(room, turn_budget=args.turn_budget)
    controlled_agents = (
        set(room.wings) if args.agent == "both" else {args.agent}
    )

    while not round_is_over(engine):
        for agent in room.wings:
            state = engine.agents[agent]
            if state.escaped or state.turns_left == 0:
                continue

            print("\n" + "=" * 72)
            print(render_observation(engine, agent))
            if agent in controlled_agents:
                output = input(f"\nAgent {agent}> ")
            else:
                print(f"\nAgent {agent}> wait()")
                output = "wait()"
            outcome = engine.step(agent, output)
            print(outcome.result)

    escaped = [agent for agent, state in engine.agents.items() if state.escaped]
    print("\nRound over. Escaped:", ", ".join(escaped) if escaped else "nobody")


def round_is_over(engine: RoomEngine) -> bool:
    return all(
        state.escaped or state.turns_left == 0 for state in engine.agents.values()
    )


if __name__ == "__main__":
    main()
