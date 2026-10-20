#!/usr/bin/env python3
"""Run a two-agent MotiveRooms episode with Azure OpenAI (joint steps)."""

from __future__ import annotations

import argparse
import json
import os
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
from motiverooms.experiment import digest, run_episode
from motiverooms.telemetry import RequestBudget
from motiverooms.actions import Action, parse_action
from motiverooms.llm_coord import choose_action_with_checks, decision_order, with_partner_pick
from motiverooms.llm_policy import LLMAgentPolicy
from motiverooms.narrative import narrate_turn
from motiverooms.schema import AgentId


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--room",
        type=Path,
        default=ROOT / "rooms/examples/shared_mat.yaml",
    )
    parser.add_argument("--turn-budget", type=int, default=14)
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
    parser.add_argument("--coaching", action="store_true", help="Add explicit sharing/withholding strategy instructions.")
    parser.add_argument("--model-a", help="Deployment for policy 1 (defaults to AZURE_OPENAI_MODEL).")
    parser.add_argument("--model-b", help="Deployment for policy 2 (defaults to AZURE_OPENAI_MODEL).")
    parser.add_argument("--framing-a", choices=("collaborative", "mixed", "competitive", "work_alone", "single"))
    parser.add_argument("--framing-b", choices=("collaborative", "mixed", "competitive", "work_alone", "single"))
    parser.add_argument("--swap-seats", action="store_true", help="Swap policy 1 and policy 2 between physical seats A/B, including model and framing.")
    parser.add_argument("--max-steps", type=int, default=20)
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
    if args.max_steps <= 0 or args.turn_budget <= 0:
        parser.error("--max-steps and --turn-budget must be positive")
    return args


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
        committed[agent] = parse_action(action_line)
    return actions


def main() -> int:
    args = parse_args()

    preview = AzureOpenAISettings.preview_from_env()
    specs = policy_specs(args, preview.model)
    print(
        f"Protocol: decisions={args.decision_mode} | mode={args.mode} | "
        f"shared_escape={args.shared_escape} | partner_actions={args.partner_actions} | "
        f"coaching={args.coaching} | swap_seats={args.swap_seats}",
    )
    print(f"  Observations: {args.observation_version}")
    for agent, (identity, model, framing) in specs.items():
        print(f"  Seat {agent}: {identity} | model={model} | framing={framing}")

    if args.dry_run:
        settings = AzureOpenAISettings.preview_from_env()
        if settings.api_key == "(unset)":
            print("Azure OpenAI dry-run (set AZURE_OPENAI_API_KEY in .env to call the API).")
        else:
            print("Azure OpenAI configuration OK.")
        print(f"  base_url: {mask_endpoint(settings.base_url)}")
        print(f"  model:    {settings.model}")
        print(f"  room:     {args.room}")
        return 0

    try:
        settings = AzureOpenAISettings.from_env()
    except AzureOpenAIConfigError as error:
        print(error, file=sys.stderr)
        return 1

    env = MotiveRoomsEnv.from_yaml(
        args.room,
        turn_budget=args.turn_budget,
        mode=args.mode,
        shared_escape=args.shared_escape,
        partner_actions=args.partner_actions,
        observation_version=args.observation_version,
    )
    policies = {
        agent: LLMAgentPolicy(
            agent=agent,
            client=AzureResponsesClient(replace(settings, model=specs[agent][1])),
            framing=specs[agent][2],
            mode=args.mode,
            shared_room=env.room.is_shared,
            shared_escape=args.shared_escape,
            partner_actions=args.partner_actions,
            coaching=args.coaching,
            repair_parse=args.decision_mode == "assisted",
        )
        for agent in env.agent_ids
    }
    budget = RequestBudget(
        max_calls=args.max_api_calls, max_tokens=args.max_tokens, max_cost_usd=args.max_cost_usd,
        input_usd_per_million=args.input_price, output_usd_per_million=args.output_price,
    )
    for policy in policies.values():
        policy.client.budget = budget

    llm_mode = "simultaneous, no action repair" if args.decision_mode == "simultaneous" else "assisted (partner picks and action repair)"
    print(
        f"Room: {env.room.id} | "
        f"joint-step rollouts | {llm_mode}"
    )
    print(f"Agents: {', '.join(env.agent_ids)}\n")

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
    result = run_episode(
        env, policies, path=args.output, max_steps=args.max_steps,
        metadata=dict(
            room_metadata(args.room, env.room),
            protocol=vars(args) | {"room": str(args.room), "output": str(args.output)},
            policies={a: dict(identity=specs[a][0], model=specs[a][1], framing=specs[a][2]) for a in env.agent_ids},
            model_settings=dict(temperature=settings.temperature, max_output_tokens=settings.max_output_tokens,
                                timeout_seconds=settings.timeout_seconds, endpoint=mask_endpoint(settings.base_url)),
            prices=dict(input_usd_per_million=args.input_price, output_usd_per_million=args.output_price),
        ),
        chooser=lambda p, o, e: request_actions(p, o, e, env_step=e.env_step,
                                              parallel=args.decision_mode == "simultaneous"),
        on_step=report_step,
    )
    print(f"Saved rollout: {args.output}")
    step = env.env_step
    if result["end"]["reason"] in {"api_error", "policy_error", "budget_exceeded"}:
        print(f"Stopped: {result['end']['reason']} ({result['end']['error_type']})")
        return 1

    escaped = [a for a in env.agent_ids if env.engine.agents[a].escaped]
    if not env.round_over:
        print(f"Truncated after {step} step(s); no terminal rewards. Escaped: {escaped or 'none'}")
        return 2
    print(f"Terminal rewards ({args.mode}): {env.rewards()}")
    if env.room.is_shared and env.engine.shared_escape == "race":
        if len(escaped) == 1:
            print(f"Finished after {step} step(s). Winner: Agent {escaped[0]}")
            return 0
        print(f"Finished after {step} step(s). No winner (escaped: {escaped or 'none'})")
        return 2
    print(f"Finished after {step} step(s). Escaped: {escaped or 'none'}")
    return 0 if len(escaped) == len(env.agent_ids) else 2


if __name__ == "__main__":
    raise SystemExit(main())
