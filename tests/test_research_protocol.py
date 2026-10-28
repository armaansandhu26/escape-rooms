"""Regression checks for experimental information boundaries and incentives."""

from threading import Barrier

import pytest

from motiverooms.env import MotiveRoomsEnv
from motiverooms.llm_policy import LLMAgentPolicy
from scripts.rollout_llm import parse_args, policy_specs, request_actions
from scripts import rollout_llm

SHARED = "rooms/examples/shared_mat.yaml"
SPLIT = "rooms/examples/office_storage.yaml"


class RecordingClient:
    def __init__(self, response: str, barrier: Barrier | None = None):
        self.response = response
        self.barrier = barrier
        self.inputs: list[str] = []

    def complete(self, *, instructions: str, user_input: str) -> str:
        self.inputs.append(user_input)
        if self.barrier:
            self.barrier.wait(timeout=5)
        return self.response


def test_default_decisions_are_independent_and_conflicts_cost_turns():
    env = MotiveRoomsEnv.from_yaml(SHARED, turn_budget=8)
    env.step_joint({"A": "enter(4721, filing_cabinet)", "B": "wait()"})
    observations = env.observe_all()
    barrier = Barrier(2)
    clients = {agent: RecordingClient("take(brass_key)", barrier) for agent in env.agent_ids}
    policies = {agent: LLMAgentPolicy(agent, client, action_interface="text") for agent, client in clients.items()}
    actions = request_actions(policies, observations, env, env_step=env.env_step)
    # Both calls must finish before either action reaches the engine.
    assert env.env_step == 1
    for agent, client in clients.items():
        assert client.inputs == [observations[agent]]
        assert "PARTNER PICK THIS STEP" not in client.inputs[0]
    outcomes = env.step_joint(actions)
    assert outcomes["A"].valid
    assert not outcomes["B"].valid
    assert all(state.turns_left == 6 for state in env.engine.agents.values())


def test_malformed_model_output_costs_a_turn_without_retry():
    env = MotiveRoomsEnv.from_yaml(SHARED, turn_budget=2)
    client = RecordingClient("I cannot decide.")
    policy = LLMAgentPolicy("A", client, action_interface="text")
    _, action = policy.choose_action(env.observe("A"))
    outcomes = env.step_joint({"A": action, "B": "wait()"})
    assert len(client.inputs) == 1
    assert not outcomes["A"].valid
    assert env.engine.agents["A"].turns_left == 1


def test_assisted_condition_exposes_partner_pick_explicitly():
    env = MotiveRoomsEnv.from_yaml(SHARED)
    clients = {agent: RecordingClient("wait()") for agent in env.agent_ids}
    policies = {agent: LLMAgentPolicy(agent, client, action_interface="text") for agent, client in clients.items()}
    request_actions(policies, env.observe_all(), env, env_step=0, parallel=False)
    assert "PARTNER PICK THIS STEP" not in clients["A"].inputs[0]
    assert "Agent A: wait()" in clients["B"].inputs[0]


@pytest.mark.parametrize("room", [SHARED, SPLIT])
def test_private_actions_do_not_leak_code_arguments_but_messages_remain_visible(room):
    env = MotiveRoomsEnv.from_yaml(room)
    target = "desk_drawer" if room == SHARED else "toolbox"
    env.step_joint({"A": f"enter(3906, {target})", "B": "wait()"})
    assert "3906" not in env.observe("B")
    assert "PARTNER LAST STEP" not in env.observe("B")
    env.partner_actions = "public"
    assert f"enter(3906, {target})" in env.observe("B")
    env.partner_actions = "private"
    env.step_joint({"A": "say(The code is 3906)", "B": "wait()"})
    assert "A: The code is 3906" in env.observe("B")


@pytest.mark.parametrize("shared_escape", ["together", "race"])
def test_reward_mode_and_framing_do_not_change_transitions(shared_escape):
    snapshots = []
    rewards = {}
    for mode in ("collaborative", "mixed", "competitive"):
        for framing in ("collaborative", "mixed", "competitive"):
            env = MotiveRoomsEnv.from_yaml(
                SHARED, turn_budget=5, mode=mode, framing=framing,
                shared_escape=shared_escape,
            )
            env.engine.agents["A"].bag.add("brass_key")
            outcomes = env.step_joint({"A": "use(brass_key, exit)", "B": "wait()"})
            snapshots.append((outcomes, env.engine.agents, env.engine.opened_nodes))
            assert env.round_over
            rewards[mode] = env.rewards()
    assert all(snapshot == snapshots[0] for snapshot in snapshots)
    if shared_escape == "race":
        assert rewards == {
            "collaborative": {"A": 14.0, "B": 14.0},
            "mixed": {"A": 14.0, "B": 0.0},
            "competitive": {"A": 14.0, "B": -14.0},
        }
    else:
        assert rewards["competitive"] == {"A": 0.0, "B": 0.0}
        assert rewards["collaborative"] == {"A": 28.0, "B": 28.0}


def test_competitive_framing_does_not_implicitly_enable_race():
    env = MotiveRoomsEnv.from_yaml(SHARED, framing="competitive", mode="competitive")
    assert env.engine.shared_escape == "together"
    with pytest.raises(ValueError, match="round end"):
        env.rewards()


def test_seat_swap_moves_policy_identity_model_and_framing_together():
    flags = ["--model-a", "deployment-1", "--model-b", "deployment-2",
             "--framing-a", "collaborative", "--framing-b", "competitive"]
    normal = policy_specs(parse_args(flags), "default")
    swapped = policy_specs(parse_args(flags + ["--swap-seats"]), "default")
    assert normal["A"] == swapped["B"] == ("policy_1", "deployment-1", "collaborative")
    assert normal["B"] == swapped["A"] == ("policy_2", "deployment-2", "competitive")
    assert parse_args([]).decision_mode == "simultaneous"
    assert parse_args(["--parallel-llm"]).decision_mode == "simultaneous"


@pytest.mark.parametrize("max_steps", [1, 2])
def test_cli_strict_failures_finish_or_truncate_without_network(monkeypatch, capsys, max_steps):
    clients = []

    def fake_client(settings, budget=None):
        client = RecordingClient("malformed response")
        clients.append(client)
        return client

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(rollout_llm, "AzureResponsesClient", fake_client)
    monkeypatch.setattr(rollout_llm, "parse_args", lambda: parse_args([
        "--turn-budget", "2", "--max-steps", str(max_steps), "--action-interface", "text", "--team-mode", "fixed",
    ]))
    assert rollout_llm.main() == 2
    output = capsys.readouterr().out
    assert "(FAIL)" in output
    assert all(len(client.inputs) == max_steps for client in clients)
    if max_steps == 1:
        assert "Truncated" in output
        assert "Terminal rewards" not in output
    else:
        assert "Terminal rewards (collaborative): {'A': 0.0, 'B': 0.0}" in output
