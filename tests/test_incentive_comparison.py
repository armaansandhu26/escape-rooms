import json

from motiverooms.azure_openai import AzureOpenAISettings
from motiverooms.evaluation import load_suite
from motiverooms.experiment import read_events, replay
from scripts.evaluate import execute_plan, make_plan


def test_incentive_pairs_alternate_order_and_keep_rooms_and_partner_fixed(tmp_path, monkeypatch):
    cases = load_suite("rooms/diagnostics/manifest.json")
    config = dict(split="development", families=["private_handoffs"], seeds=[0],
                  conditions=["honest"], both_seats=True, repeats=2, modes=["mixed", "collaborative"],
                  observation_version="v3", action_interface="text")
    plan = make_plan(config, cases)
    assert len(plan) == 8
    assert [p[1]["mode"] for p in plan[:4]] == ["mixed", "collaborative", "collaborative", "mixed"]

    class Settings:
        @staticmethod
        def from_env():
            return AzureOpenAISettings("test", "https://example.com/openai/v1/", "test")

    class Client:
        def __init__(self, settings, budget):
            self.settings = settings

        def complete(self, *, instructions, user_input):
            return "wait()"

    monkeypatch.setattr("scripts.evaluate.AzureOpenAISettings", Settings)
    records = execute_plan(config, plan, tmp_path / "paired", client_factory=Client)
    assert len(records) == 8
    for first, second in zip(records[::2], records[1::2]):
        events = [read_events(r["path"]) for r in (first, second)]
        starts = [e[0] for e in events]
        assert starts[0]["room_sha256"] == starts[1]["room_sha256"]
        assert starts[0]["metadata"]["script_policy_versions"] == starts[1]["metadata"]["script_policy_versions"]
        states = [[e["after"] for e in episode if e["type"] == "joint_step"] for episode in events]
        assert states[0] == states[1]
        for r, start in zip((first, second), starts):
            mode = r["metadata"]["mode"]
            assert start["config"]["mode"] == start["metadata"]["framing"] == mode
            assert all(f"Actual scoring mode: {mode}." in prompt for prompt in start["prompts"].values())
            replay(r["path"])
    summary = json.loads((tmp_path / "paired/condition_summary.json").read_text())
    assert {s["mode"] for s in summary} == {"mixed", "collaborative"}
    assert "Completed matched collaborative/mixed pairs: **4**" in (tmp_path / "paired/summary.md").read_text()
