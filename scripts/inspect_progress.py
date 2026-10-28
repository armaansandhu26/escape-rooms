#!/usr/bin/env python3
"""Inspect torch/puzzle milestones in saved runs without making model calls."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def inspect_log(path: Path) -> dict:
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    start = next(event for event in events if event["type"] == "start")
    steps = [event for event in events if event["type"] == "joint_step"]
    end = next((event for event in reversed(events) if event["type"] == "end"), None)
    node_ids = {node["id"] for node in start["room"]["nodes"]}
    item_ids = set(start["room"]["items"])
    milestones = []
    for label, id, field in (
        ("Desk drawer opened", "desk_drawer", "opened_nodes"),
        ("Batteries collected", "batteries", "taken_items"),
        ("Torch powered", "unpowered_torch", "opened_nodes"),
        ("Working torch collected", "uv_torch", "taken_items"),
        ("Painting illuminated", "painting", "opened_nodes"),
        ("Display case opened", "display_case", "opened_nodes"),
    ):
        if id in node_ids | item_ids:
            first = next((step["step"] for step in steps if id in step["after"][field]), None)
            milestones.append((label, first))
    if "painting_inscription" in node_ids:
        first = next((step["step"] for step in steps if any(
            "painting_inscription" in agent["read_codes"] for agent in step["after"]["agents"].values())), None)
        milestones.append(("Painting clue read", first))
    escaped = next((step["step"] for step in steps if all(
        agent["escaped"] for agent in step["after"]["agents"].values())), None)
    milestones.append(("Escaped", escaped))
    inspections = Counter((agent, action) for step in steps for agent, action in step["actions"].items()
                          if action.startswith("inspect("))
    invalid = sum(not outcome["valid"] for step in steps for outcome in step["outcomes"].values())
    wrong = sum(outcome["valid"] and outcome["result"] == "wrong"
                for step in steps for outcome in step["outcomes"].values())
    return dict(path=str(path), reason=end["reason"] if end else "incomplete log", steps=len(steps),
                turn_budget=start["config"]["turn_budget"], milestones=milestones,
                invalid_actions=invalid, wrong_attempts=wrong,
                repeated_inspections=sum(count - 1 for count in inspections.values()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="A saved episode JSONL or a batch directory.")
    args = parser.parse_args()
    paths = sorted(args.path.glob("*.jsonl")) if args.path.is_dir() else [args.path]
    if not paths or any(not path.is_file() for path in paths):
        parser.error("No episode logs found")
    for path in paths:
        report = inspect_log(path)
        print(f"\n{path.name}: {report['reason']} after {report['steps']}/{report['turn_budget']} ticks")
        for label, step in report["milestones"]:
            print(f"  {label}: {'tick ' + str(step) if step is not None else 'not completed'}")
        print(f"  Rejected actions: {report['invalid_actions']} | Wrong attempts: {report['wrong_attempts']}")
        print(f"  Repeat inspections beyond each worker's first: {report['repeated_inspections']} (may be justified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
