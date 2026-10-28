#!/usr/bin/env python3
"""Run an orchestrated MotiveRooms episode with Azure OpenAI (joint steps)."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_dotenv_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv_file(ROOT / ".env")

from motiverooms.azure_openai import (
    AzureOpenAIConfigError,
    AzureOpenAISettings,
    AzureResponsesClient,
    mask_endpoint,
)
from motiverooms.env import MotiveRoomsEnv
from motiverooms.connected_generator import ConnectedRoomsConfig, generate_connected_episode
from motiverooms.exploration_generator import generate_exploration_episode
from motiverooms.experiment import digest, run_episode
from motiverooms.telemetry import RequestBudget
from motiverooms.actions import Action, ActionParseError, parse_action
from motiverooms.llm_coord import choose_action_with_checks, decision_order, with_partner_pick
from motiverooms.llm_policy import LLMAgentPolicy
from motiverooms.narrative import narrate_turn
from motiverooms.orchestration import Orchestrator, OrchestratorLimits, run_orchestrated_episode
from motiverooms.schema import AgentId, load_room


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--room",
        type=Path,
        help="Use a saved room instead of generating a fresh episode.",
    )
    parser.add_argument("--room-source", choices=("exploration", "connected"),
                        help="Generated puzzle family (default: exploration).")
    parser.add_argument("--exploration-version", type=int, choices=(1, 2),
                        help="Exploration generator version (default: 2; 1 reproduces the original pilot).")
    parser.add_argument("--num-rooms", type=int, help="Physical rooms per generated episode (default: 3).")
    parser.add_argument("--seed", type=int, help="Reproduce a generated episode; omitted selects a fresh recorded seed.")
    parser.add_argument("--turn-budget", type=int, help="Per-agent turns (default: scaled to the generated reference, or 14 for files).")
    parser.add_argument(
        "--framing",
        default=None,
        choices=("collaborative", "mixed", "competitive", "work_alone", "single"),
        help="Prompt framing; defaults to the reward mode. Does not change mechanics.",
    )
    parser.add_argument("--mode", choices=("collaborative", "mixed", "competitive"), default="collaborative")
    parser.add_argument("--shared-escape", choices=("together", "race"), default="together")
    parser.add_argument("--partner-actions", choices=("private", "public"), default="private")
    parser.add_argument("--observation-version", choices=("v1", "v2", "v3"), default="v3")
    parser.add_argument("--action-interface", choices=("tools", "text"), default="tools",
                        help="Native function calls (default) or legacy last-line text actions.")
    parser.add_argument("--team-mode", choices=("orchestrated", "fixed"), default="orchestrated")
    parser.add_argument("--min-subagents", type=int, default=2, help="Minimum orchestrated worker count (default: 2).")
    parser.add_argument("--max-subagents", type=int, default=4, help="Upper bound; the orchestrator chooses the actual count (default: 4).")
    parser.add_argument("--orchestrator-context-version", choices=("v1", "v2", "v3", "v4", "v5"), default="v5")
    parser.add_argument("--orchestrator-check-in-interval", type=int, default=5,
                        help="V5: check on a worker after this many silent ticks.")
    parser.add_argument("--orchestrator-model", help="Controller deployment (defaults to AZURE_OPENAI_MODEL).")
    parser.add_argument("--orchestrator-max-output-tokens", type=int, default=2048)
    parser.add_argument("--coaching", action="store_true", help="Add explicit sharing/withholding strategy instructions.")
    parser.add_argument("--model-a", help="Deployment for policy 1 (defaults to AZURE_OPENAI_MODEL).")
    parser.add_argument("--model-b", help="Deployment for policy 2 (defaults to AZURE_OPENAI_MODEL).")
    parser.add_argument("--framing-a", choices=("collaborative", "mixed", "competitive", "work_alone", "single"))
    parser.add_argument("--framing-b", choices=("collaborative", "mixed", "competitive", "work_alone", "single"))
    parser.add_argument("--swap-seats", action="store_true", help="Swap policy 1 and policy 2 between physical seats A/B, including model and framing.")
    parser.add_argument("--max-steps", type=int, help="Joint-step cutoff (default: the turn budget).")
    parser.add_argument("--output", type=Path, help="JSONL path (default: a new file under runs/).")
    parser.add_argument("--max-api-calls", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=250_000)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--input-price", type=float, help="USD per million input tokens; user-supplied deployment price.")
    parser.add_argument("--output-price", type=float, help="USD per million output tokens.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without calling Azure.",
    )
    decisions = parser.add_mutually_exclusive_group()
    decisions.add_argument("--decision-mode", choices=("simultaneous", "assisted"), default="simultaneous")
    decisions.add_argument(
        "--parallel-llm",
        dest="decision_mode",
        action="store_const",
        const="simultaneous",
        help="Alias for --decision-mode simultaneous (the default).",
    )
    args = parser.parse_args(argv)
    positive_options = {
        "--max-steps": args.max_steps,
        "--turn-budget": args.turn_budget,
        "--num-rooms": args.num_rooms,
        "--max-subagents": args.max_subagents,
        "--min-subagents": args.min_subagents,
        "--orchestrator-check-in-interval": args.orchestrator_check_in_interval,
        "--orchestrator-max-output-tokens": args.orchestrator_max_output_tokens,
    }
    for option, value in positive_options.items():
        if value is not None and value <= 0:
            parser.error(f"{option} must be positive")
    if args.min_subagents > args.max_subagents:
        parser.error("--min-subagents must not exceed --max-subagents")
    if args.room and any(value is not None for value in (
        args.num_rooms, args.seed, args.room_source, args.exploration_version,
    )):
        parser.error("Generation options apply to generated episodes; omit --room")
    if args.room_source == "connected" and args.exploration_version is not None:
        parser.error("--exploration-version requires --room-source exploration")
    if args.team_mode == "orchestrated" and args.decision_mode != "simultaneous":
        parser.error("Orchestrated teams use simultaneous decisions; select --team-mode fixed for assisted runs")
    return args


def load_episode(args):
    """Resolve an episode once, including on previews; never call a provider."""
    if args.room:
        room = load_room(args.room)
        return room, room_metadata(args.room, room)
    seed = args.seed if args.seed is not None else secrets.randbits(63)
    settings = ConnectedRoomsConfig(args.num_rooms or 3)
    if args.room_source == "connected":
        episode = generate_connected_episode(seed, settings)
    else:
        episode = generate_exploration_episode(seed, settings, version=args.exploration_version or 2)
    return episode.room, episode.metadata


def policy_specs(args: argparse.Namespace, default_model: str) -> dict[AgentId, tuple[str, str, str]]:
    """Map stable policy identities to seats; room asymmetries stay fixed."""
    framing = args.framing or args.mode
    specs = [
        ("policy_1", args.model_a or default_model, args.framing_a or framing),
        ("policy_2", args.model_b or default_model, args.framing_b or framing),
    ]
    if args.swap_seats:
        specs.reverse()
    return dict(zip(("A", "B"), specs))


def room_metadata(path, room):
    """Attach evaluator-only diagnostic facts when running a generated YAML directly."""
    path = Path(path).resolve()
    manifest = path.parent.parent / "manifest.json"
    if manifest.is_file():
        for case in json.loads(manifest.read_text())["cases"]:
            if (manifest.parent / case["room_file"]).resolve() == path:
                if digest(room.to_dict()) != case["room_sha256"]:
                    raise ValueError("Room differs from its frozen diagnostic manifest")
                return case
    return dict(seed=None, family="file", source_file=str(path), handoffs=[])


def request_actions(
    policies: dict[AgentId, LLMAgentPolicy],
    observations: dict[AgentId, str],
    engine: MotiveRoomsEnv,
    *,
    env_step: int,
    parallel: bool = True,
) -> dict[AgentId, str]:
    if not observations:
        return {}
    if parallel:
        actions: dict[AgentId, str] = {}

        def work(agent: AgentId) -> tuple[AgentId, str]:
            print(f"  … Agent {agent}: calling Azure", flush=True)
            _, action_line = policies[agent].choose_action(observations[agent])
            return agent, action_line

        with ThreadPoolExecutor(max_workers=len(observations)) as pool:
            futures = [pool.submit(work, agent) for agent in observations]
            for future in as_completed(futures):
                agent, action = future.result()
                actions[agent] = action
        return actions

    def log_llm_wait(agent: AgentId, attempt: int) -> None:
        suffix = f" (retry {attempt})" if attempt > 1 else ""
        print(f"  … Agent {agent}: calling Azure{suffix}", flush=True)

    order = decision_order(tuple(observations), env_step)
    actions: dict[AgentId, str] = {}
    committed: dict[AgentId, Action] = {}
    room_engine = engine.engine
    for index, agent in enumerate(order):
        observation = observations[agent]
        if index > 0:
            partner = order[index - 1]
            observation = with_partner_pick(observation, partner, actions[partner])
        action_line = choose_action_with_checks(
            policies[agent],
            observation,
            room_engine,
            agent,
            committed,
            on_llm_wait=log_llm_wait,
        )
        actions[agent] = action_line
        try:
            committed[agent] = parse_action(action_line)
        except ActionParseError:
            pass  # Exhausted malformed attempts still reach the engine and cost a turn.
    return actions


def main() -> int:
    args = parse_args()
    room, episode_metadata = load_episode(args)
    args.turn_budget = args.turn_budget or episode_metadata.get("recommended_turn_budget", 14)
    args.max_steps = args.max_steps or args.turn_budget

    preview = AzureOpenAISettings.preview_from_env()
    specs = policy_specs(args, preview.model)
    print(
        f"Protocol: decisions={args.decision_mode} | mode={args.mode} | "
        f"shared_escape={args.shared_escape} | partner_actions={args.partner_actions} | "
        f"coaching={args.coaching} | swap_seats={args.swap_seats}",
    )
    print(f"  Observations: {args.observation_version}")
    print(f"  Action interface: {args.action_interface}")
    print(f"  Team mode: {args.team_mode} | subagents: min={args.min_subagents}, max={args.max_subagents}")
    if args.team_mode == "orchestrated":
        print(f"  Orchestration protocol: {args.orchestrator_context_version}")
    if args.team_mode == "orchestrated":
        print("  Orchestrator chooses the team at execution time; previews make no model calls.")
    print(f"  Episode: {room.id} | physical rooms={len(room.locations) or 1} | seed={episode_metadata.get('seed')}")
    if "generator" in episode_metadata:
        print(f"  Generator: {episode_metadata['generator']} v{episode_metadata['generator_version']}")
    print(f"  Turn budget: {args.turn_budget} | max steps: {args.max_steps}")
    if args.team_mode == "fixed":
        for agent, (identity, model, framing) in specs.items():
            print(f"  Seat {agent}: {identity} | model={model} | framing={framing}")
    else:
        print(f"  Orchestrator model: {args.orchestrator_model or preview.model}")
        print(f"  Default worker model: {preview.model}")

    if args.dry_run:
        settings = AzureOpenAISettings.preview_from_env()
        if settings.api_key == "(unset)":
            print("Azure OpenAI dry-run (set AZURE_OPENAI_API_KEY in .env to call the API).")
        else:
            print("Azure OpenAI configuration OK.")
        print(f"  base_url: {mask_endpoint(settings.base_url)}")
        print(f"  model:    {settings.model}")
        print(f"  room:     {args.room or 'generated ' + (args.room_source or 'exploration') + ' rooms'}")
        return 0

    try:
        settings = AzureOpenAISettings.from_env()
    except AzureOpenAIConfigError as error:
        print(error, file=sys.stderr)
        return 1

    env_options = dict(turn_budget=args.turn_budget, mode=args.mode, shared_escape=args.shared_escape,
                       partner_actions=args.partner_actions, observation_version=args.observation_version)
    budget = RequestBudget(
        max_calls=args.max_api_calls, max_tokens=args.max_tokens, max_cost_usd=args.max_cost_usd,
        input_usd_per_million=args.input_price, output_usd_per_million=args.output_price,
    )

    def worker_factory(agent, env, assignment=""):
        index = tuple(env.agent_ids).index(agent)
        spec = list(specs.values())[index] if index < len(specs) else (
            f"policy_{index + 1}", settings.model, args.framing or args.mode)
        return LLMAgentPolicy(
            agent=agent, client=AzureResponsesClient(replace(settings, model=spec[1]), budget=budget),
            framing=spec[2], mode=args.mode, shared_room=env.room.is_shared,
            shared_escape=args.shared_escape, partner_actions=args.partner_actions,
            coaching=args.coaching, repair_parse=args.decision_mode == "assisted",
            action_interface=args.action_interface, navigation=bool(room.locations),
            solo=len(env.agent_ids) == 1, coordination=env.coordination,
            teammates=env.agent_ids, assignment=assignment,
            orchestration_protocol=args.orchestrator_context_version,
            history_turns=3 if env.coordination and args.orchestrator_context_version in {"v4", "v5"} else None,
            progress_memory=env.coordination and args.orchestrator_context_version in {"v4", "v5"},
        )

    llm_mode = "simultaneous, no action repair" if args.decision_mode == "simultaneous" else "assisted (partner picks and action repair)"
    print(
        f"Room: {room.id} | "
        f"joint-step rollouts | {llm_mode}"
    )

    def report_step(env, actions, outcomes):
        for agent, outcome in outcomes.items():
            tag, line = narrate_turn(
                env.room,
                agent,
                actions[agent],
                outcome,
                engine=env.engine,
            )
            status = "ok" if outcome.valid else "FAIL"
            print(f"  Step {env.env_step} | Agent {agent} | {actions[agent]}")
            print(f"    [{tag}] ({status}) {line}")
        print()
    if args.output is None:
        from uuid import uuid4
        args.output = ROOT / "runs" / f"episode_{uuid4().hex}.jsonl"
    metadata = dict(
        episode_metadata,
        protocol=vars(args) | {"room": str(args.room) if args.room else None,
                              "output": str(args.output), "seed": episode_metadata.get("seed"),
                              "num_rooms": len(room.locations) or None},
        action_interface=args.action_interface,
        model_settings=dict(temperature=settings.temperature, max_output_tokens=settings.max_output_tokens,
                            timeout_seconds=settings.timeout_seconds, endpoint=mask_endpoint(settings.base_url)),
        prices=dict(input_usd_per_million=args.input_price, output_usd_per_million=args.output_price),
    )
    if args.team_mode == "orchestrated":
        controller_settings = replace(settings, model=args.orchestrator_model or settings.model,
                                      max_output_tokens=args.orchestrator_max_output_tokens)
        controller = Orchestrator(AzureResponsesClient(controller_settings, budget=budget),
                                  OrchestratorLimits(max_subagents=args.max_subagents, min_subagents=args.min_subagents),
                                  context_version=args.orchestrator_context_version,
                                  history_turns=1 if args.orchestrator_context_version in {"v4", "v5"} else None,
                                  check_in_interval=args.orchestrator_check_in_interval)
        def report_team(plan):
            print(f"Orchestrator created {len(plan.agents)} subagents: {', '.join(plan.agents)}", flush=True)
            print(f"  Reason: {plan.rationale}", flush=True)
        result = run_orchestrated_episode(room, controller, worker_factory, path=args.output,
                                          metadata=metadata, max_steps=args.max_steps,
                                          on_step=report_step, on_team_created=report_team, **env_options)
    else:
        env = MotiveRoomsEnv(room, **env_options)
        policies = {agent: worker_factory(agent, env) for agent in env.agent_ids}
        metadata["policies"] = {a: dict(identity=specs[a][0], model=specs[a][1], framing=specs[a][2])
                                for a in env.agent_ids}
        result = run_episode(
            env, policies, path=args.output, max_steps=args.max_steps, metadata=metadata,
            chooser=lambda p, o, e: request_actions(p, o, e, env_step=e.env_step,
                                                  parallel=args.decision_mode == "simultaneous"),
            on_step=report_step,
        )
    print(f"Saved rollout: {args.output}")
    final_state = result["end"]["state"]
    step = final_state["env_step"]
    if result["end"]["reason"] in {"api_error", "policy_error", "budget_exceeded"}:
        print(f"Stopped: {result['end']['reason']} ({result['end']['error_type']})")
        return 1

    agents = final_state["agents"]
    escaped = [agent for agent, state in agents.items() if state["escaped"]]
    if result["end"]["reason"] == "max_steps":
        print(f"Truncated after {step} step(s); no terminal rewards. Escaped: {escaped or 'none'}")
        return 2
    print(f"Terminal rewards ({args.mode}): {result['end']['rewards']}")
    if room.is_shared and args.shared_escape == "race":
        if len(escaped) == 1:
            print(f"Finished after {step} step(s). Winner: Agent {escaped[0]}")
            return 0
        print(f"Finished after {step} step(s). No winner (escaped: {escaped or 'none'})")
        return 2
    print(f"Finished after {step} step(s). Escaped: {escaped or 'none'}")
    return 0 if len(escaped) == len(agents) else 2


if __name__ == "__main__":
    raise SystemExit(main())
