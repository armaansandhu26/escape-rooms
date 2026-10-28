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
from motiverooms.connected_generator import ConnectedRoomsConfig, generate_connected_episode
from motiverooms.exploration_generator import generate_exploration_episode
from motiverooms.torch_puzzle import generate_torch_episode
from motiverooms.env import MotiveRoomsEnv
from motiverooms.evaluation import load_suite, solo_projection, summarize
from motiverooms.experiment import replay, run_episode
from motiverooms.llm_policy import LLMAgentPolicy
from motiverooms.memory import validate_history_turns
from motiverooms.telemetry import RequestBudget
from motiverooms.orchestration import (Orchestrator, OrchestratorLimits, ScriptedOrchestrator,
                                      reference_for_team, run_orchestrated_episode)


def load_cases(config, suite):
    """Keep generated episode maps separate from frozen diagnostic suites."""
    source = config.get("room_source", "suite")
    if source == "suite":
        return load_suite(suite)
    if source not in {"connected", "exploration", "torch_dependency"}:
        raise ValueError(f"Unknown room source: {source}")
    settings = ConnectedRoomsConfig(config.get("num_rooms", 3))
    supported = {"solo", "reference"} if source == "torch_dependency" else {"self_play", "reference"}
    if set(config["conditions"]) - supported:
        raise ValueError(f"{source} evaluations support {', '.join(sorted(supported))} conditions")
    seeds = config.get("seeds")
    if seeds is None:
        count = config.get("episodes", 1)
        if type(count) is not int or count < 1:
            raise ValueError("episodes must be a positive integer")
        base = config.get("seed", 0)
        seeds = range(base, base + count)
    if source == "torch_dependency":
        if settings.num_rooms != 1 or config.get("team_mode", "fixed") != "fixed":
            raise ValueError("torch_dependency requires num_rooms: 1 and team_mode: fixed")
        cases = [generate_torch_episode(seed) for seed in seeds]
    elif source == "exploration":
        cases = [generate_exploration_episode(seed, settings, version=config.get("exploration_version", 2),
                                             torch_affordance=config.get("torch_affordance"))
                 for seed in seeds]
    else:
        cases = [generate_connected_episode(seed, settings) for seed in seeds]
    return [(case.room, case.metadata) for case in cases]


def team_mode(config):
    mode = config.get("team_mode", "orchestrated" if config.get("room_source") in {"connected", "exploration"} else "fixed")
    if mode not in {"orchestrated", "fixed"}:
        raise ValueError(f"Unknown team mode: {mode}")
    return mode


def make_plan(config, cases, offline=False):
    plan = []
    for option in ("turn_budget", "max_steps", "orchestrator_check_in_interval"):
        if option in config and (type(config[option]) is not int or config[option] < 1):
            raise ValueError(f"{option} must be a positive integer")
    roster_mode = team_mode(config)
    if type(config.get("worker_broadcast", True)) is not bool:
        raise ValueError("worker_broadcast must be a boolean")
    if not config.get("worker_broadcast", True) and roster_mode != "orchestrated":
        raise ValueError("worker_broadcast: false requires an orchestrated team")
    OrchestratorLimits(max_subagents=config.get("max_subagents", 4),
                       min_subagents=config.get("min_subagents", 2))
    for setting in ("worker_history_turns", "orchestrator_history_turns"):
        validate_history_turns(config.get(setting))
    if config.get("orchestrator_context_version", "v1") not in {"v1", "v2", "v3", "v4", "v5"}:
        raise ValueError("orchestrator_context_version must be v1, v2, v3, v4 or v5")
    if type(config.get("worker_progress_memory", False)) is not bool:
        raise ValueError("worker_progress_memory must be a boolean")
    action_interface = config.get("action_interface", "tools")
    if action_interface not in {"tools", "text"}:
        raise ValueError(f"Unknown action interface: {action_interface}")
    instruction = config.get("communication_instruction", "original")
    if instruction not in {"original", "explained"}:
        raise ValueError(f"Unknown communication instruction: {instruction}")
    versions = config.get("observation_versions", [config.get("observation_version", "v3")])
    if not versions or set(versions) - {"v1", "v2", "v3"}:
        raise ValueError("Observation versions must be v1, v2 or v3")
    modes = config.get("modes", [config.get("mode", "mixed")])
    if not modes or set(modes) - {"mixed", "collaborative", "competitive"}:
        raise ValueError("Unknown reward modes")
    conditions = ["reference"] if offline else config["conditions"]
    communication_modes = config.get("communication_modes", ["enabled"])
    if (not communication_modes or len(set(communication_modes)) != len(communication_modes)
            or set(communication_modes) - {"enabled", "disabled"}):
        raise ValueError("Communication modes must be unique enabled/disabled values")
    if roster_mode == "orchestrated":
        if "disabled" in communication_modes:
            raise ValueError("Orchestration requires enabled communication; use team_mode: fixed for communication-off controls")
        if set(conditions) - {"self_play", "reference"}:
            raise ValueError("Orchestrated evaluations support self_play or offline reference controls")
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
                            for communication in (list(reversed(communication_modes)) if repeat % 2 else communication_modes):
                                plan.append((room, dict(meta, observation_version=version, mode=mode,
                                                        action_interface=action_interface, team_mode=roster_mode,
                                                        communication=communication,
                                                        worker_broadcast=config.get("worker_broadcast", True),
                                                        worker_progress_memory=config.get("worker_progress_memory", False),
                                                        communication_instruction=instruction), condition, seat, repeat))
    return plan


def execute_plan(config, plan, output, *, offline=False, client_factory=AzureResponsesClient):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "plan.json").write_text(json.dumps([
        dict(room_id=room.id, condition=condition, seat=seat, repeat=repeat,
             mode=meta["mode"], observation_version=meta["observation_version"],
             action_interface=meta["action_interface"],
             num_rooms=meta.get("num_rooms", 1), team_mode=meta["team_mode"],
             worker_broadcast=meta.get("worker_broadcast", True),
             worker_progress_memory=meta.get("worker_progress_memory", False),
             communication=meta["communication"], communication_instruction=meta["communication_instruction"])
        for room, meta, condition, seat, repeat in plan
    ], indent=2) + "\n")
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
        env = MotiveRoomsEnv(room, turn_budget=config.get("turn_budget", meta["recommended_turn_budget"]),
                            mode=meta.get("mode", config.get("mode", "mixed")), partner_actions="private", shared_escape="together",
                            observation_version=meta.get("observation_version", "v3"),
                            communication=meta.get("communication", "enabled"),
                            worker_broadcast=meta.get("worker_broadcast", True))
        def worker_factory(agent, worker_env, assignment=""):
            if offline:
                reference_meta = reference_for_team(meta, worker_env.agent_ids) if worker_env.coordination and room.is_shared else meta
                return ScriptedPolicy(agent, worker_env, reference_meta,
                                      "reference" if worker_env.communication == "enabled" or room.locations else "silent")
            if worker_env.coordination or condition in {"solo", "self_play"} or agent == seat:
                model = config.get("model") or settings.model
                client = client_factory(replace(settings, model=model), budget=budget)
                return LLMAgentPolicy(
                    agent, client, framing=config.get("framing") or worker_env.mode,
                    mode=worker_env.mode, shared_room=room.is_shared, shared_escape=worker_env.shared_escape,
                    communication=worker_env.communication,
                    communication_instruction=meta.get("communication_instruction", "original"),
                    action_interface=meta.get("action_interface", config.get("action_interface", "tools")),
                    solo=len(worker_env.agent_ids) == 1, navigation=bool(room.locations),
                    coordination=worker_env.coordination, teammates=worker_env.agent_ids, assignment=assignment,
                    history_turns=config.get("worker_history_turns"),
                    controller_observes_chat=config.get("orchestrator_observe_team_chat", False),
                    worker_broadcast=worker_env.worker_broadcast,
                    progress_memory=meta.get("worker_progress_memory", False),
                    orchestration_protocol=config.get("orchestrator_context_version", "v1"),
                )
            return ScriptedPolicy(agent, worker_env, meta, condition)

        policies = {} if meta["team_mode"] == "orchestrated" else {
            agent: worker_factory(agent, env) for agent in env.agent_ids}
        metadata = dict(meta, source_room_id=source_room.id, evaluated_seat=seat, condition=condition,
                        repeat=repeat, mode=env.mode, communication=env.communication,
                        turn_budget=env.turn_budget, max_steps=config.get("max_steps", env.turn_budget),
                        experiment=config.get("experiment"),
                        framing=config.get("framing") or env.mode,
                        model=None if offline else config.get("model") or settings.model,
                        temperature=None if offline else settings.temperature,
                        max_output_tokens=None if offline else settings.max_output_tokens,
                        note="Room seed controls generation; repeat is a trial index, not an API sampling seed.")
        metadata["protocol"] = dict(decisions="simultaneous", coaching=False, repair_parse=False,
                                    action_interface=metadata["action_interface"], team_mode=meta["team_mode"],
                                    min_subagents=config.get("min_subagents", 2),
                                    max_subagents=config.get("max_subagents", 4),
                                    orchestrator_model=config.get("orchestrator_model"),
                                    orchestrator_context_version=config.get("orchestrator_context_version", "v1"),
                                    orchestrator_observe_team_chat=config.get("orchestrator_observe_team_chat", False),
                                    orchestrator_history_turns=config.get("orchestrator_history_turns"),
                                    orchestrator_check_in_interval=config.get("orchestrator_check_in_interval", 5),
                                    worker_history_turns=config.get("worker_history_turns"),
                                    worker_progress_memory=meta.get("worker_progress_memory", False),
                                    partner_actions="private", shared_escape="together", communication=env.communication,
                                    worker_broadcast=env.worker_broadcast)
        metadata["budget"] = config.get("budget", {})
        metadata["policies"] = {a: ("azure" if isinstance(p, LLMAgentPolicy) else p.kind) for a, p in policies.items()}
        metadata["script_policy_versions"] = {a: p.policy_version for a, p in policies.items() if isinstance(p, ScriptedPolicy)}
        if condition == "solo":
            metadata["handoffs"] = []
            metadata["oracle"] = {"A": meta["oracle"][seat]}
        path = output / f"{index:05d}_{room.id}_{condition}_{seat}_{repeat}_{env.mode}_{env.observation_version}_{env.communication}.jsonl"
        shown_tasks = {}
        shown_exit_briefings = {}
        def show_step(current_env, actions, outcomes):
            if config.get("progress", False):
                if current_env.coordination and config.get("orchestrator_context_version") in {"v4", "v5"}:
                    decision = controller.events[-1]
                    current = decision["phase"] == "coordinate" and decision["step"] == current_env.env_step
                    if current and "check_in" in decision["input"]:
                        print(f"  Manager check-in, tick {current_env.env_step}: "
                              f"{', '.join(decision['input']['check_in']['reasons'])}", flush=True)
                    briefings = decision.get("result", {}).get("exit_briefings", []) if current else []
                    for briefing in briefings:
                        key = briefing["node_id"]
                        facts = (briefing["room_id"], briefing["code"], briefing["status"])
                        if shown_exit_briefings.get(key) != facts:
                            print(f"  Manager exit briefing → all workers: {key} in "
                                  f"{briefing['room_id'] or 'unknown room'} | code: {briefing['code'] or 'unknown'}", flush=True)
                            shown_exit_briefings[key] = facts
                    updates = decision.get("result", {}).get("updated_assignments", {}) if current else {}
                    for agent, task in updates.items():
                        if shown_tasks.get(agent) != task:
                            print(f"  Controller, tick {current_env.env_step} → {agent}: {task}", flush=True)
                            shown_tasks[agent] = task
                choices = "; ".join(f"{agent}: {action} [{'wrong' if outcomes[agent].result == 'wrong' else 'ok' if outcomes[agent].valid else 'rejected'}]"
                                    for agent, action in actions.items())
                print(f"  Tick {current_env.env_step}/{current_env.turn_budget}: {choices}", flush=True)

        def show_team(plan):
            if config.get("progress", False):
                print(f"  Orchestrator created {len(plan.agents)} workers: {', '.join(plan.agents)}", flush=True)
                if config.get("orchestrator_context_version") in {"v4", "v5"}:
                    print(f"  Team rationale: {plan.rationale}", flush=True)
                    for agent, task in plan.tasks.items():
                        print(f"  Initial assignment → {agent}: {task}", flush=True)
                        shown_tasks[agent] = task

        if meta["team_mode"] == "orchestrated":
            limits = OrchestratorLimits(max_subagents=config.get("max_subagents", 4),
                                        min_subagents=config.get("min_subagents", 2))
            controller_options = dict(context_version=config.get("orchestrator_context_version", "v1"),
                                      observe_team_chat=config.get("orchestrator_observe_team_chat", False),
                                      history_turns=config.get("orchestrator_history_turns"),
                                      check_in_interval=config.get("orchestrator_check_in_interval", 5))
            if offline:
                controller = ScriptedOrchestrator(limits, **controller_options)
            else:
                controller_settings = replace(settings, model=config.get("orchestrator_model") or config.get("model") or settings.model,
                                              max_output_tokens=config.get("orchestrator_max_output_tokens", 1024))
                controller = Orchestrator(client_factory(controller_settings, budget=budget), limits, **controller_options)
            if config.get("progress", False):
                print(f"Episode {index + 1}/{len(plan)}: planning team for {meta.get('num_rooms', 1)} rooms…", flush=True)
            result = run_orchestrated_episode(
                room, controller, worker_factory, path=path, metadata=metadata,
                turn_budget=env.turn_budget, mode=env.mode, shared_escape=env.shared_escape,
                partner_actions=env.partner_actions, communication=env.communication,
                worker_broadcast=env.worker_broadcast,
                observation_version=env.observation_version, max_steps=config.get("max_steps", env.turn_budget),
                on_team_created=show_team, on_step=show_step)
            metadata = result["events"][0]["metadata"]
        else:
            result = run_episode(env, policies, path=path, metadata=metadata,
                                 max_steps=config.get("max_steps", env.turn_budget), on_step=show_step)
        replay(path)
        records.append(dict(metadata=metadata, metrics=result["metrics"], path=str(path)))
        summarize(records, output)  # Keep a usable partial report if a later run stops.
        print(f"{index + 1}/{len(plan)} {room.id} {condition}/{seat}/{env.mode}/{env.observation_version}/{env.communication}: {result['end']['reason']}", flush=True)
        if result["end"]["reason"] == "budget_exceeded":
            break
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/eval_exploration_varied_manager_v5.yaml")
    parser.add_argument("--suite", type=Path, default=ROOT / "rooms/diagnostics/manifest.json")
    parser.add_argument("--split", choices=("development", "heldout", "all"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--action-interface", choices=("tools", "text"),
                        help="Override the evaluation's native tools / legacy text interface.")
    parser.add_argument("--team-mode", choices=("orchestrated", "fixed"))
    parser.add_argument("--max-subagents", type=int)
    parser.add_argument("--min-subagents", type=int)
    parser.add_argument("--orchestrator-check-in-interval", type=int,
                        help="V5: check on a worker after this many silent ticks (default: 5).")
    parser.add_argument("--num-rooms", type=int, help="Override the physical room count in a generated evaluation.")
    parser.add_argument("--turn-budget", type=int, help="Override per-agent turns for each episode.")
    parser.add_argument("--max-steps", type=int, help="Override the joint-step cutoff.")
    parser.add_argument("--limit-episodes", type=int, help="Run/preview only the first N planned episodes for a smoke test.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true", help="Run the planned Azure calls.")
    group.add_argument("--offline", action="store_true", help="Execute only scripted reference trajectories; no Azure calls.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.team_mode is not None:
        config["team_mode"] = args.team_mode
    if args.max_subagents is not None:
        config["max_subagents"] = args.max_subagents
    if args.min_subagents is not None:
        config["min_subagents"] = args.min_subagents
    if args.orchestrator_check_in_interval is not None:
        config["orchestrator_check_in_interval"] = args.orchestrator_check_in_interval
    if args.num_rooms is not None:
        if config.get("room_source") not in {"connected", "exploration", "torch_dependency"}:
            parser.error("--num-rooms requires a generated room source")
        config["num_rooms"] = args.num_rooms
    for option in ("turn_budget", "max_steps"):
        value = getattr(args, option)
        if value is not None:
            config[option] = value
    if args.action_interface:
        config["action_interface"] = args.action_interface
    if args.split:
        config["split"] = args.split
    if args.offline and "communication_modes" not in config and config.get("room_source", "suite") == "suite":
        config["seeds"] = None
    load_dotenv_file(ROOT / ".env")
    plan = make_plan(config, load_cases(config, args.suite), args.offline)
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
    print(f"Communication: {sorted({p[1]['communication'] for p in plan})}")
    print(f"Communication instructions: {config.get('communication_instruction', 'original')}")
    print(f"Worker broadcast: {config.get('worker_broadcast', True)}")
    print(f"Worker progress memory: {config.get('worker_progress_memory', False)}")
    print(f"Action interface: {config.get('action_interface', 'tools')}")
    print(f"Room source: {config.get('room_source', 'suite')} | rooms per episode: {config.get('num_rooms', 'suite-defined')}")
    print(f"Turn budget: {config.get('turn_budget', 'per-room recommendation')} | "
          f"max steps: {config.get('max_steps', 'turn budget')}")
    print(f"Team mode: {team_mode(config)} | subagents: "
          f"min={config.get('min_subagents', 2)}, max={config.get('max_subagents', 4)}")
    if team_mode(config) == "orchestrated":
        if config.get("orchestrator_context_version") == "v5":
            print(f"Manager checks: on reports or after {config.get('orchestrator_check_in_interval', 5)} silent ticks per worker")
        print(f"Orchestration protocol: {config.get('orchestrator_context_version', 'v1')} | "
              f"public chat to controller: {config.get('orchestrator_observe_team_chat', False)} | "
              f"history turns: worker={config.get('worker_history_turns', 'all')}, "
              f"controller={config.get('orchestrator_history_turns', 'all')}")
    print(f"Session budget: {config.get('budget', {})}")
    if not args.execute and not args.offline:
        print("Preview only. --execute runs Azure; --offline runs reference controls without API calls.")
        return
    output = args.output or ROOT / "runs" / f"batch_{uuid4().hex}"
    records = execute_plan(config, plan, output, offline=args.offline)
    print(f"Recorded {len(records)}/{len(plan)} episodes. Report: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
