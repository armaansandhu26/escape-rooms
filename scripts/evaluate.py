#!/usr/bin/env python3
"""Preview a baseline matrix, run offline controls, or explicitly execute Azure evaluations."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from scripts.rollout_llm import load_dotenv_file
from motiverooms.azure_openai import AzureOpenAISettings, AzureResponsesClient
from motiverooms.benchmark_policies import ScriptedPolicy
from motiverooms.env import MotiveRoomsEnv
from motiverooms.evaluation import load_suite, solo_projection, summarize
from motiverooms.experiment import replay, run_episode
from motiverooms.llm_policy import LLMAgentPolicy
from motiverooms.telemetry import RequestBudget


def make_plan(config, cases, offline=False):
    plan = []
    versions = config.get("observation_versions", [config.get("observation_version", "v3")])
    if not versions or set(versions) - {"v1", "v2", "v3"}:
        raise ValueError("Observation versions must be v1, v2 or v3")
    modes = config.get("modes", [config.get("mode", "mixed")])
    if not modes or set(modes) - {"mixed", "collaborative", "competitive"}:
        raise ValueError("Unknown reward modes")
    conditions = ["reference"] if offline else config["conditions"]
    if set(conditions) - {"reference", "solo", "honest", "silent", "deceptive", "self_play"}:
        raise ValueError("Unknown evaluation condition")
    for room, meta in cases:
        if config["split"] != "all" and meta["split"] != config["split"]:
            continue
        if config.get("seeds") is not None and meta["seed"] not in config["seeds"]:
            continue
        if config.get("families") and meta["family"] not in config["families"]:
            continue
        for condition in conditions:
            seats = ["A", "B"] if config.get("both_seats", True) and condition != "reference" else ["A"]
            if condition == "solo" and room.is_shared:
                seats = ["A"]
            for seat in seats:
                for repeat in range(config.get("repeats", 1)):
                    for version in versions:
                        ordered_modes = list(reversed(modes)) if repeat % 2 else modes
                        for mode in ordered_modes:
                            plan.append((room, dict(meta, observation_version=version, mode=mode), condition, seat, repeat))
    return plan


def execute_plan(config, plan, output, *, offline=False, client_factory=AzureResponsesClient):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    budget = RequestBudget(**config.get("budget", {}))
    settings = None if offline else AzureOpenAISettings.from_env()
    if settings is not None:
        settings = replace(
            settings,
            temperature=config.get("temperature", settings.temperature),
            max_output_tokens=config.get("max_output_tokens", settings.max_output_tokens),
        )
    records = []
    for index, (source_room, meta, condition, seat, repeat) in enumerate(plan):
        room = solo_projection(source_room, seat) if condition == "solo" else source_room
        env = MotiveRoomsEnv(room, turn_budget=meta["recommended_turn_budget"],
                            mode=meta.get("mode", config.get("mode", "mixed")), partner_actions="private", shared_escape="together",
                            observation_version=meta.get("observation_version", "v3"))
        policies = {}
        for agent in env.agent_ids:
            if offline:
                policies[agent] = ScriptedPolicy(agent, env, meta, "reference")
            elif condition in {"solo", "self_play"} or agent == seat:
                model = config.get("model") or settings.model
                client = client_factory(replace(settings, model=model), budget=budget)
                policies[agent] = LLMAgentPolicy(
                    agent, client, framing=config.get("framing") or env.mode,
                    mode=env.mode, shared_room=room.is_shared, shared_escape=env.shared_escape,
                )
            else:
                policies[agent] = ScriptedPolicy(agent, env, meta, condition)
        metadata = dict(meta, source_room_id=source_room.id, evaluated_seat=seat, condition=condition,
                        repeat=repeat, mode=env.mode,
                        framing=config.get("framing") or env.mode,
                        model=None if offline else config.get("model") or settings.model,
                        temperature=None if offline else settings.temperature,
                        max_output_tokens=None if offline else settings.max_output_tokens,
                        note="Room seed controls generation; repeat is a trial index, not an API sampling seed.")
        metadata["protocol"] = dict(decisions="simultaneous", coaching=False, repair_parse=False,
                                    partner_actions="private", shared_escape="together")
        metadata["budget"] = config.get("budget", {})
        metadata["policies"] = {a: ("azure" if isinstance(p, LLMAgentPolicy) else p.kind) for a, p in policies.items()}
        metadata["script_policy_versions"] = {a: p.policy_version for a, p in policies.items() if isinstance(p, ScriptedPolicy)}
        if condition == "solo":
            metadata["handoffs"] = []
            metadata["oracle"] = {"A": meta["oracle"][seat]}
        path = output / f"{index:05d}_{room.id}_{condition}_{seat}_{repeat}_{env.mode}_{env.observation_version}.jsonl"
        result = run_episode(env, policies, path=path, metadata=metadata,
                             max_steps=config.get("max_steps", 50))
        replay(path)
        records.append(dict(metadata=metadata, metrics=result["metrics"], path=str(path)))
        summarize(records, output)  # Keep a usable partial report if a later run stops.
        print(f"{index + 1}/{len(plan)} {room.id} {condition}/{seat}/{env.mode}/{env.observation_version}: {result['end']['reason']}", flush=True)
        if result["end"]["reason"] == "budget_exceeded":
            break
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/eval_pilot.yaml")
    parser.add_argument("--suite", type=Path, default=ROOT / "rooms/diagnostics/manifest.json")
    parser.add_argument("--split", choices=("development", "heldout", "all"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit-episodes", type=int, help="Run/preview only the first N planned episodes for a smoke test.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true", help="Run the planned Azure calls.")
    group.add_argument("--offline", action="store_true", help="Execute only scripted reference trajectories; no Azure calls.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.split:
        config["split"] = args.split
    if args.offline:
        config["seeds"] = None
    load_dotenv_file(ROOT / ".env")
    plan = make_plan(config, load_suite(args.suite), args.offline)
    if args.limit_episodes is not None:
        if args.limit_episodes < 1:
            parser.error("--limit-episodes must be positive")
        plan = plan[:args.limit_episodes]
    if not plan:
        parser.error("No rooms match this evaluation plan")
    print(f"Planned episodes: {len(plan)} | offline={args.offline} | split={config['split']}")
    print(f"Conditions: {sorted({p[2] for p in plan})}")
    print(f"Observation versions: {sorted({p[1]['observation_version'] for p in plan})}")
    print(f"Reward modes: {sorted({p[1]['mode'] for p in plan})}")
    print(f"Session budget: {config.get('budget', {})}")
    if not args.execute and not args.offline:
        print("Preview only. --execute runs Azure; --offline runs reference controls without API calls.")
        return
    output = args.output or ROOT / "runs" / f"batch_{uuid4().hex}"
    records = execute_plan(config, plan, output, offline=args.offline)
    print(f"Recorded {len(records)}/{len(plan)} episodes. Report: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
