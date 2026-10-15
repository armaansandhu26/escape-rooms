#!/usr/bin/env python3
"""Run a two-agent MotiveRooms episode with Azure OpenAI (joint steps)."""

from __future__ import annotations

import argparse
import os
import sys
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
from motiverooms.actions import Action, parse_action
from motiverooms.llm_coord import choose_action_with_checks, decision_order, with_partner_pick
from motiverooms.llm_policy import LLMAgentPolicy
from motiverooms.narrative import narrate_turn
from motiverooms.schema import AgentId


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--room",
        type=Path,
        default=ROOT / "rooms/examples/shared_mat.yaml",
    )
    parser.add_argument("--turn-budget", type=int, default=14)
    parser.add_argument(
        "--framing",
        default="collaborative",
        choices=("collaborative", "mixed", "competitive"),
        help="Incentive prompt (collaborative, mixed, competitive).",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without calling Azure.",
    )
    parser.add_argument(
        "--parallel-llm",
        action="store_true",
        help="Call the LLM for all agents at once (often duplicates actions).",
    )
    return parser.parse_args()


def request_actions(
    policies: dict[AgentId, LLMAgentPolicy],
    observations: dict[AgentId, str],
    engine: MotiveRoomsEnv,
    *,
    env_step: int,
    parallel: bool = False,
) -> dict[AgentId, str]:
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

    client = AzureResponsesClient(settings)
    env = MotiveRoomsEnv.from_yaml(
        args.room,
        turn_budget=args.turn_budget,
        framing=args.framing,
    )
    policies = {
        agent: LLMAgentPolicy(agent=agent, client=client, framing=args.framing)
        for agent in env.agent_ids
    }

    llm_mode = "parallel LLM" if args.parallel_llm else "staggered LLM (second agent sees partner pick)"
    print(
        f"Room: {env.room.id} | framing: {args.framing} | model: {settings.model} | "
        f"joint-step rollouts | {llm_mode}"
    )
    print(f"Agents: {', '.join(env.agent_ids)}\n")

    step = 0
    while not env.round_over and step < args.max_steps:
        active = env.active_agents()
        observations = {agent: env.observe(agent) for agent in active}
        print(f"--- Step {env.env_step + 1} (LLM calls: {len(active)}) ---")
        try:
            actions = request_actions(
                {agent: policies[agent] for agent in active},
                observations,
                env,
                env_step=env.env_step,
                parallel=args.parallel_llm,
            )
        except Exception as error:
            print(f"LLM error: {error}", file=sys.stderr)
            return 1

        outcomes = env.step_joint(actions)
        for agent in active:
            outcome = outcomes[agent]
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
        step += 1

    escaped = [a for a in env.agent_ids if env.engine.agents[a].escaped]
    if env.engine.shared_escape == "race":
        if len(escaped) == 1:
            print(f"Finished after {step} step(s). Winner: Agent {escaped[0]}")
            return 0
        print(f"Finished after {step} step(s). No winner (escaped: {escaped or 'none'})")
        return 2
    print(f"Finished after {step} step(s). Escaped: {escaped or 'none'}")
    return 0 if escaped else 2


if __name__ == "__main__":
    raise SystemExit(main())
