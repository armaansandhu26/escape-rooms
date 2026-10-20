#!/usr/bin/env python3
"""Generate and validate a frozen diagnostic suite, without model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motiverooms.engine import RoomEngine
from motiverooms.env import MotiveRoomsEnv
from motiverooms.generator import FAMILIES, generate_room
from motiverooms.schema import dump_room, load_room


def validate_case(case):
    budget = case.metadata["recommended_turn_budget"]
    for agent, actions in case.metadata["solo_plans"].items():
        engine = RoomEngine(case.room, budget)
        for action in actions:
            outcome = engine.step(agent, action)
            if not outcome.valid or outcome.result == "wrong":
                raise ValueError(f"Invalid solo reference: {case.room.id}: {action}")
        if not engine.agents[agent].escaped:
            raise ValueError(f"Unsolvable solo route: {case.room.id}")
    env = MotiveRoomsEnv(case.room, turn_budget=budget)
    for tick in case.metadata["reference"]:
        if env.round_over:
            break
        outcomes = env.step_joint({a: tick.get(a, "wait()") for a in env.active_agents()})
        if any(not o.valid or o.result == "wrong" for o in outcomes.values()):
            raise ValueError(f"Invalid joint reference: {case.room.id}: {outcomes}")
    if not all(s.escaped for s in env.engine.agents.values()):
        raise ValueError(f"Joint reference did not escape: {case.room.id}")
    return env.env_step


def build_suite(output: Path, count: int = 10):
    if count < 1:
        raise ValueError("count must be positive")
    output.mkdir(parents=True, exist_ok=True)
    cases, pending = [], []
    for split in ("development", "heldout"):
        for family in FAMILIES:
            for seed in range(count):
                profile = "balanced" if family == "parallel_branches" else (
                    "zero" if seed % 5 == 4 else "asymmetric" if seed % 5 == 3 else "balanced"
                )
                case = generate_room(family, seed, split=split, profile=profile)
                ticks = validate_case(case)
                case.metadata["reference_joint_steps"] = ticks
                path = output / split / f"{case.room.id}.yaml"
                pending.append((path, case.room))
                payload = json.dumps(case.room.to_dict(), sort_keys=True).encode()
                cases.append(dict(room_file=str(path.relative_to(output)),
                                  room_sha256=hashlib.sha256(payload).hexdigest(), **case.metadata))
    manifest = dict(schema_version=1, generator_version=1, cases=cases)
    path = output / "manifest.json"
    content = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != content:
        raise ValueError("Existing manifest differs; use a new output directory for a new suite")
    for room_path, room in pending:
        if room_path.exists() and load_room(room_path).to_dict() != room.to_dict():
            raise ValueError(f"Existing room differs: {room_path}; use a new output directory")
    for room_path, room in pending:
        if not room_path.exists():
            room_path.parent.mkdir(parents=True, exist_ok=True)
            dump_room(room, room_path)
    path.write_text(content)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "rooms/diagnostics")
    parser.add_argument("--count", type=int, default=10, help="Variants per family per split.")
    args = parser.parse_args()
    manifest = build_suite(args.output, args.count)
    print(f"Validated {len(manifest['cases'])} rooms: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
