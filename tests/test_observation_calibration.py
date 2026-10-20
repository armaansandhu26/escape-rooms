import json

from motiverooms.benchmark_policies import ScriptedPolicy
from motiverooms.env import MotiveRoomsEnv
from motiverooms.evaluation import solo_projection, summarize
from motiverooms.experiment import replay, run_episode
from motiverooms.generator import generate_room
from motiverooms.render import render_observation
from scripts.evaluate import make_plan


def test_v2_distinguishes_clues_inventory_and_completed_locks():
    env = MotiveRoomsEnv.from_yaml("rooms/examples/shared_mat.yaml")
    env.step_joint({"A": "enter(4721, filing_cabinet)", "B": "wait()"})
    obs = env.observe("A")
    assert "wall_poster (type=clue)" in obs
    assert "Read with inspect(wall_poster); this is not a takeable item." in obs
    assert "filing_cabinet (type=container)" in obs
    assert "STATUS: OPEN. Unlocking is complete" in obs
    assert "- brass_key (item; source=filing_cabinet)" in obs
    env.step_joint({"A": "take(brass_key)", "B": "wait()"})
    env.step_joint({"A": "use(brass_key, filing_cabinet)", "B": "wait()"})
    obs = env.observe("A")
    assert "YOUR BAG: brass key (item_id=brass_key)" in obs
    assert "YOUR LAST ACTION: use(brass_key, filing_cabinet)" in obs
    assert "LAST RESULT: That node is already open." in obs
    assert "YOUR RECENT ACTIONS (last 3)" in obs
    assert "- brass_key (item; source=filing_cabinet)" not in obs
    assert "PARTNER LAST STEP" not in obs


def test_v2_does_not_reveal_private_codes_or_hidden_available_items():
    case = generate_room("private_handoffs", 0)
    env = MotiveRoomsEnv(case.room)
    handoff = next(h for h in case.metadata["handoffs"] if h["holder"] == "B")
    env.step_joint({"A": "wait()", "B": handoff["acquire"][0]})
    assert case.room.codes[handoff["resource"]] not in env.observe("A")
    assert case.room.codes[handoff["resource"]] in env.observe("B")
    shared = MotiveRoomsEnv.from_yaml("rooms/examples/shared_mat.yaml")
    available = shared.observe("A").split("AVAILABLE ITEMS (takeable objects only)")[1].split("YOUR LAST ACTION")[0]
    assert "brass_key" not in available


def test_observation_versions_preserve_transitions_and_legacy_replay(tmp_path):
    case = generate_room("parallel_branches", 0)
    results = []
    for version in ("v1", "v2", "v3"):
        env = MotiveRoomsEnv(case.room, observation_version=version)
        policies = {a: ScriptedPolicy(a, env, case.metadata, "reference") for a in env.agent_ids}
        path = tmp_path / f"{version}.jsonl"
        result = run_episode(env, policies, path=path)
        assert replay(path) == result["metrics"]
        results.append(result)
        if version == "v1":
            events = result["events"]
            del events[0]["config"]["observation_version"]
            path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
            assert replay(path) == result["metrics"]
    first_steps = [e for e in results[0]["events"] if e["type"] == "joint_step"]
    second_steps = [e for e in results[1]["events"] if e["type"] == "joint_step"]
    third_steps = [e for e in results[2]["events"] if e["type"] == "joint_step"]
    assert [e["after"] for e in first_steps] == [e["after"] for e in second_steps]
    assert [e["after"] for e in second_steps] == [e["after"] for e in third_steps]
    assert first_steps[0]["observations"] != second_steps[0]["observations"]


def test_paired_calibration_plan_and_reports_keep_versions_separate(tmp_path):
    case = generate_room("parallel_branches", 0)
    config = dict(split="development", conditions=["solo"], observation_versions=["v1", "v2"],
                  seeds=[0], both_seats=False)
    plan = make_plan(config, [(case.room, case.metadata)])
    assert [meta["observation_version"] for _, meta, _, _, _ in plan] == ["v1", "v2"]
    assert all(condition == "solo" for _, _, condition, _, _ in plan)
    records = []
    for _, meta, condition, seat, repeat in plan:
        env = MotiveRoomsEnv(case.room, observation_version=meta["observation_version"])
        policies = {a: ScriptedPolicy(a, env, case.metadata, "reference") for a in env.agent_ids}
        result = run_episode(env, policies)
        metadata = dict(meta, condition=condition, source_room_id=case.room.id,
                        evaluated_seat=seat, repeat=repeat, mode="mixed")
        records.append(dict(metadata=metadata, metrics=result["metrics"], path=f"{meta['observation_version']}.jsonl"))
    summary = summarize(records, tmp_path)
    assert len(summary) == 2
    assert {s["observation_version"] for s in summary} == {"v1", "v2"}
    assert "Completed matched v1/v2 pairs: **1**" in (tmp_path / "summary.md").read_text()


def test_v3_explains_lock_methods_without_revealing_codes():
    env = MotiveRoomsEnv.from_yaml("rooms/examples/shared_mat.yaml", observation_version="v3")
    obs = env.observe("A")
    assert "CODE LOCK: enter(<4_digit_code>, desk_drawer)" in obs
    assert "ITEM LOCK: use(<item_id>, exit)" in obs
    assert "Digits inside object/item IDs are identifier text, not evidence of a lock code" in obs
    assert env.room.codes["drawer"] not in obs
    assert "LAST RESULT: The round has not started." in obs
    env.step_joint({"A": "enter(3906, desk_drawer)", "B": "wait()"})
    obs = env.observe("A")
    assert "CODE LOCK: enter(<4_digit_code>, desk_drawer)" not in obs
    assert "STATUS: OPEN. Unlocking is complete" in obs
    legacy = render_observation(env.engine, "A", observation_version="v2")
    assert "LOCK AND ID RULES" not in legacy
    assert "ITEM LOCK:" not in legacy


def test_report_counts_wrong_attempts_as_unsuccessful_without_changing_engine_semantics(tmp_path):
    case = generate_room("parallel_branches", 0)
    room = solo_projection(case.room, "A")
    exit_id = next(n.id for n in room.nodes if n.kind == "door")
    records = []
    for version in ("v2", "v3"):
        env = MotiveRoomsEnv(room, turn_budget=2, observation_version=version)

        class Policy:
            def __init__(self):
                self.actions = iter(["take(not_an_item)", f"enter(9999, {exit_id})"])

            def choose_action(self, observation):
                action = next(self.actions)
                return action, action

        result = run_episode(env, {"A": Policy()})
        stats = result["metrics"]["per_agent"]["A"]
        assert stats["invalid_actions"] == stats["wrong_attempts"] == 1
        assert stats["unsuccessful_actions"] == 2
        assert stats["unsuccessful_action_rate"] == 1.0
        steps = [e for e in result["events"] if e["type"] == "joint_step"]
        assert steps[1]["outcomes"]["A"]["valid"]  # Preserve valid-but-wrong semantics.
        meta = dict(case.metadata, condition="solo", source_room_id=case.room.id,
                    evaluated_seat="A", repeat=0, mode="mixed", observation_version=version)
        records.append(dict(metadata=meta, metrics=result["metrics"], path=f"{version}.jsonl"))
    summarize(records, tmp_path)
    text = (tmp_path / "summary.md").read_text()
    assert "LLM wrong / actions" in text
    assert "2/2 (100.0%)" in text
    assert "Completed matched v2/v3 pairs: **1**" in text
