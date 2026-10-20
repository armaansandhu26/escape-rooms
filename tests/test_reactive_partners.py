import pytest

from motiverooms.benchmark_policies import ScriptedPolicy
from motiverooms.env import MotiveRoomsEnv
from motiverooms.generator import generate_room


@pytest.mark.parametrize("kind", ["honest", "silent"])
@pytest.mark.parametrize("seat", ["A", "B"])
def test_partner_switches_branches_after_other_agent_takes_its_item(kind, seat):
    case = generate_room("parallel_branches", 0)
    env = MotiveRoomsEnv(case.room, turn_budget=20)
    partner = "B" if seat == "A" else "A"
    script = ScriptedPolicy(seat, env, case.metadata, kind)
    # The other seat executes the script's designated branch and keeps its token.
    for tick in case.metadata["reference"][:3]:
        env.step_joint({seat: "wait()", partner: tick[seat]})
    stolen = case.metadata["reference"][2][seat]
    assert stolen.startswith("take(")
    actions = []
    for _ in range(4):
        _, action = script.choose_action(env.observe(seat))
        actions.append(action)
        outcomes = env.step_joint({seat: action, partner: "wait()"})
        assert outcomes[seat].valid
        assert outcomes[seat].result != "wrong"
    assert any(a.startswith("take(") for a in actions)
    assert stolen not in actions
    if kind == "silent":
        assert not any(a.startswith("say(") for a in actions)


@pytest.mark.parametrize("family", ["parallel_branches", "private_handoffs", "costly_help"])
@pytest.mark.parametrize("kind", ["honest", "silent"])
def test_reactive_partner_pair_solves_diagnostic_rooms(family, kind):
    case = generate_room(family, 0)
    env = MotiveRoomsEnv(case.room, turn_budget=case.metadata["recommended_turn_budget"])
    policies = {a: ScriptedPolicy(a, env, case.metadata, kind) for a in env.agent_ids}
    while not env.round_over:
        actions = {a: policies[a].choose_action(env.observe(a))[1] for a in env.active_agents()}
        outcomes = env.step_joint(actions)
        assert all(o.result != "wrong" for o in outcomes.values())
        # Two independent policies can still choose the same object in a tick.
        assert all(o.valid or o.result in {"Someone else took it this step.", "That node is already open."} for o in outcomes.values())
    assert all(s.escaped for s in env.engine.agents.values())


def test_honest_waits_for_a_hidden_helper_clue_before_sharing_its_code():
    case = generate_room("private_handoffs", 0, split="heldout")
    env = MotiveRoomsEnv(case.room)
    script = ScriptedPolicy("A", env, case.metadata, "honest")
    handoff = next(h for h in case.metadata["handoffs"] if h["holder"] == "A")
    for index in range(len(handoff["acquire"])):
        action = script.choose_action(env.observe("A"))[1]
        assert not action.startswith("say(")
        env.step_joint({"A": action, "B": "wait()"})
    action = script.choose_action(env.observe("A"))[1]
    assert action.startswith("say(")
    assert case.room.codes[handoff["resource"]] in action
