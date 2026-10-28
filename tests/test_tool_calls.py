"""Native tool protocol, privacy, accounting and engine/replay integration."""
from copy import deepcopy
import json
from threading import Barrier
from types import SimpleNamespace

import pytest

from motiverooms.actions import ACTION_ARITY, ActionParseError, parse_action
from motiverooms.azure_openai import AzureOpenAISettings, AzureResponsesClient
from motiverooms.env import MotiveRoomsEnv
from motiverooms.experiment import replay, run_episode
from motiverooms.llm_policy import LLMAgentPolicy
from motiverooms.telemetry import BudgetExceeded, RequestBudget
from motiverooms.tool_calls import ToolCall, ToolResponse, tool_definitions
from scripts.rollout_llm import parse_args, request_actions
from scripts.evaluate import make_plan


def response(name="wait", arguments=None, call_id="call-1", **kwargs):
    return ToolResponse((dict(type="function_call", name=name,
                              arguments=json.dumps(arguments or {}), call_id=call_id),), **kwargs)


class ToolClient:
    def __init__(self, responses, barrier=None):
        self.responses = iter(responses)
        self.requests = []
        self.barrier = barrier

    def complete_tools(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if self.barrier:
            self.barrier.wait(timeout=5)
        return next(self.responses)


@pytest.mark.parametrize("name,arguments,expected", [
    ("inspect", {"node_id": "sticky_note"}, "inspect(sticky_note)"),
    ("take", {"item_id": "brass_key"}, "take(brass_key)"),
    ("enter", {"node_id": "exit_a", "code": "0042"}, "enter(0042, exit_a)"),
    ("use", {"node_id": "exit_a", "item_id": "brass_key"}, "use(brass_key, exit_a)"),
    ("give", {"item_id": "brass_key"}, "give(brass_key)"),
    ("say", {"message": "Code: 0042,\ntry it (now)"}, "say(Code: 0042, try it (now))"),
    ("write_note", {"text": "A: 0042"}, "write_note(A: 0042)"),
    ("read_notes", {}, "read_notes()"),
    ("wait", {}, "wait()"),
])
def test_native_arguments_round_trip_through_engine_format(name, arguments, expected):
    action = response(name, arguments).single_action()
    assert str(action) == expected
    assert parse_action(str(action)) == action


@pytest.mark.parametrize("name,arguments", [
    ("unknown", "{}"), ("wait", '{"extra":"x"}'),
    ("inspect", "[]"), ("inspect", "null"), ("inspect", "{}"),
    ("inspect", '{"node_id":42}'), ("inspect", '{"node_id":"brass key"}'),
    ("inspect", '{"node_id":"exit_a)\\nwait("}'),
    ("enter", '{"code":42,"node_id":"exit_a"}'),
    ("enter", '{"code":"４２","node_id":"exit_a"}'),
    ("say", '{"message":"  "}'), ("wait", "{broken"),
    ("inspect", '{"node_id":"desk","node_id":"exit_a"}'),
])
def test_malformed_arguments_are_rejected_without_coercion(name, arguments):
    with pytest.raises(ActionParseError):
        ToolCall("call-1", name, arguments).action()


def test_schemas_are_strict_static_and_filter_communication():
    enabled = tool_definitions()
    assert {t["name"] for t in enabled} == {"inspect", "take", "enter", "use", "give", "say", "write_note", "read_notes", "wait"}
    for tool in enabled:
        assert tool["strict"] is True
        assert tool["parameters"]["additionalProperties"] is False
        assert set(tool["parameters"]["required"]) == set(tool["parameters"]["properties"])
    schemas = tool_definitions(navigation=True, coordination=True) + tool_definitions(navigation=True, coordination=True, team_size=3)
    assert {tool["name"]: len(tool["parameters"]["required"]) for tool in schemas} == ACTION_ARITY
    assert {t["name"] for t in tool_definitions(communication="disabled")} == {"inspect", "take", "enter", "use", "wait"}
    enabled[0]["parameters"]["properties"].clear()
    assert tool_definitions()[0]["parameters"]["properties"]  # No shared mutable schemas.
    with pytest.raises(ActionParseError, match="Unavailable"):
        response("say", {"message": "hello"}).single_action(communication="disabled")
    assert {t["name"] for t in LLMAgentPolicy("A", None, solo=True).tools()} == {"inspect", "take", "enter", "use", "wait"}


def test_tools_are_default_and_prompts_do_not_request_text_actions():
    policy = LLMAgentPolicy("A", None)
    assert parse_args([]).action_interface == "tools"
    assert "last line" not in policy.system_prompt()
    assert "Allowed actions" not in policy.system_prompt()
    assert "exactly one" in policy.system_prompt()
    assert "last line" in LLMAgentPolicy("A", None, action_interface="text").system_prompt()
    with pytest.raises(ValueError, match="action interface"):
        LLMAgentPolicy("A", None, action_interface="bad")
    with pytest.raises(ValueError, match="action interface"):
        make_plan({"action_interface": "bad"}, [])


def test_native_private_histories_receive_only_own_results_and_replay(tmp_path):
    env = MotiveRoomsEnv.from_yaml("rooms/examples/office_storage.yaml", turn_budget=2)
    barrier = Barrier(2)
    clients = {
        "A": ToolClient([response("inspect", {"node_id": "sticky_note"}, "a1"), response(call_id="a2")], barrier),
        "B": ToolClient([response("inspect", {"node_id": "filing_tag"}, "b1"), response(call_id="b2")], barrier),
    }
    policies = {a: LLMAgentPolicy(a, c) for a, c in clients.items()}
    path = tmp_path / "tools.jsonl"
    result = run_episode(env, policies, path=path)
    assert result["end"]["reason"] == "turn_budget"
    assert replay(path) == result["metrics"]
    assert env.env_step == 2
    assert result["metrics"]["total_actions"] == 4
    for agent, client in clients.items():
        assert len(client.requests) == 2
        first, second = client.requests
        assert len(first["input_items"]) == 1
        assert "last line" not in first["input_items"][0]["content"]
        transcript = json.dumps(second["input_items"])
        assert ("4721" if agent == "A" else "2235") in transcript
        assert ("2235" if agent == "A" else "4721") not in transcript
        outputs = [item for item in second["input_items"] if item.get("type") == "function_call_output"]
        assert [item["call_id"] for item in outputs] == [agent.lower() + "1"]
        assert json.loads(outputs[0]["output"])["valid"] is True
    assert result["events"][0]["action_interfaces"] == {"A": "tools", "B": "tools"}
    assert result["events"][0]["metadata"]["action_interface"] == "tools"
    assert len([e for e in result["events"] if e["type"] == "tool_result"]) == 4


def test_native_simultaneous_conflicts_still_cost_both_agents_a_turn():
    env = MotiveRoomsEnv.from_yaml("rooms/examples/shared_mat.yaml", turn_budget=4)
    env.step_joint({"A": "enter(4721, filing_cabinet)", "B": "wait()"})
    barrier = Barrier(2)
    clients = {a: ToolClient([response("take", {"item_id": "brass_key"})], barrier) for a in env.agent_ids}
    policies = {a: LLMAgentPolicy(a, c) for a, c in clients.items()}
    actions = request_actions(policies, env.observe_all(), env, env_step=1)
    assert env.env_step == 1
    outcomes = env.step_joint(actions)
    assert outcomes["A"].valid and not outcomes["B"].valid
    assert all(s.turns_left == 2 for s in env.engine.agents.values())


def test_native_cooperative_escape_delivers_messages_items_and_terminal_results(tmp_path):
    env = MotiveRoomsEnv.from_yaml("rooms/examples/office_storage.yaml", turn_budget=8, mode="mixed")
    clients = {
        "A": ToolClient([
            response("inspect", {"node_id": "sticky_note"}, "a1"),
            response("say", {"message": "The cabinet code is 4721"}, "a2"),
            response("use", {"item_id": "brass_key", "node_id": "exit_a"}, "a3"),
        ]),
        "B": ToolClient([
            response("take", {"item_id": "brass_key"}, "b1"),
            response("give", {"item_id": "brass_key"}, "b2"),
            response("enter", {"code": "4721", "node_id": "cabinet"}, "b3"),
            response("inspect", {"node_id": "exit_card"}, "b4"),
            response("enter", {"code": "8350", "node_id": "exit_b"}, "b5"),
        ]),
    }
    policies = {a: LLMAgentPolicy(a, c, mode="mixed") for a, c in clients.items()}
    path = tmp_path / "escape.jsonl"
    result = run_episode(env, policies, path=path)
    assert result["end"]["reason"] == "success"
    assert result["end"]["rewards"] == {"A": 15.0, "B": 13.0}
    assert [len(clients[a].requests) for a in ("A", "B")] == [3, 5]
    assert result["metrics"]["total_actions"] == 8
    assert "The cabinet code is 4721" in clients["B"].requests[2]["input_items"][-1]["content"]
    outputs = [e for e in result["events"] if e["type"] == "tool_result"]
    assert len(outputs) == 8 and outputs[-1]["call_id"] == "b5"
    assert all(json.loads(e["output"])["valid"] for e in outputs)
    assert replay(path) == result["metrics"]


@pytest.mark.parametrize("bad_response", [
    ToolResponse((), text="wait()"),
    ToolResponse((*response(call_id="one").output, *response(call_id="two").output)),
    response("inspect", {"node_id": 42}),
    response(call_id=""),
    response(status="incomplete"),
])
def test_bad_calls_cost_one_turn_and_never_execute_text_or_multiple_actions(bad_response):
    env = MotiveRoomsEnv.from_yaml("rooms/examples/office_solo.yaml", turn_budget=1)
    client = ToolClient([bad_response])
    policy = LLMAgentPolicy("A", client, solo=True)
    result = run_episode(env, {"A": policy})
    assert result["end"]["reason"] == "turn_budget"
    assert result["metrics"]["per_agent"]["A"]["invalid_actions"] == 1
    assert len(client.requests) == 1
    assert not env.engine.opened_nodes
    outputs = [e for e in result["events"] if e["type"] == "tool_result"]
    assert all(json.loads(e["output"])["tool_error"] for e in outputs)


def azure_client(create, budget=None):
    client = object.__new__(AzureResponsesClient)
    client.settings = AzureOpenAISettings("not-for-logs", "https://example.com/openai/v1/", "test")
    client.budget, client.calls = budget, []
    client._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return client


def test_native_transport_records_schemas_reasoning_calls_and_usage():
    requests = []
    reasoning = {"type": "reasoning", "id": "r1", "summary": []}
    native = response().output[0]

    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        return SimpleNamespace(output_text="", output=[reasoning, native], id="resp-1", status="completed",
                               usage=dict(input_tokens=100, output_tokens=20, total_tokens=120))

    budget = RequestBudget()
    client = azure_client(create, budget)
    policy = LLMAgentPolicy("A", client)
    env = MotiveRoomsEnv.from_yaml("rooms/examples/office_solo.yaml", turn_budget=2)
    result = run_episode(env, {"A": policy})
    assert result["end"]["reason"] == "turn_budget"
    assert requests[0]["tool_choice"] == "required"
    assert requests[0]["parallel_tool_calls"] is False
    assert reasoning in requests[1]["input"]
    assert native in requests[1]["input"]
    record = client.calls[0]
    assert record["output_items"] == [reasoning, native]
    assert record["tools"] == policy.tools()
    assert "output_error" not in record
    assert "not-for-logs" not in json.dumps(result)
    assert budget.calls == 2 and budget.tokens == 240
    assert result["metrics"]["api_calls"] == 2


def test_tool_schemas_are_included_in_budget_reservations():
    client = azure_client(lambda **_: pytest.fail("Must stop before provider call"),
                          RequestBudget(max_tokens=1000))
    with pytest.raises(BudgetExceeded):
        client.complete_tools(instructions="", input_items=[], tools=tool_definitions())
    assert not client.calls


def test_tool_transport_failure_keeps_accounting_and_no_environment_transition():
    def create(**kwargs):
        raise TimeoutError("provider timeout")
    budget = RequestBudget()
    policy = LLMAgentPolicy("A", azure_client(create, budget), solo=True)
    env = MotiveRoomsEnv.from_yaml("rooms/examples/office_solo.yaml")
    result = run_episode(env, {"A": policy})
    assert result["end"]["reason"] == "api_error"
    assert env.env_step == 0
    assert result["metrics"]["api_calls"] == 1
    assert result["metrics"]["unknown_usage_calls"] == 1
    assert budget.tokens > 0 and budget.reserved_tokens == 0


def test_assisted_tools_resolve_rejected_calls_before_retrying():
    from motiverooms.llm_coord import choose_action_with_checks
    env = MotiveRoomsEnv.from_yaml("rooms/examples/office_solo.yaml")
    client = ToolClient([response("take", {"item_id": "missing"}), response(call_id="retry")])
    policy = LLMAgentPolicy("A", client, solo=True)
    action = choose_action_with_checks(policy, env.observe("A"), env.engine, "A", {})
    assert action == "wait()"
    assert len(client.requests) == 2
    outputs = [item for item in client.requests[1]["input_items"] if item.get("type") == "function_call_output"]
    assert len(outputs) == 1 and json.loads(outputs[0]["output"])["valid"] is False


def test_sdk_output_items_serialize_for_continuation():
    pytest.importorskip("openai")
    from openai.types.responses import ResponseFunctionToolCall, ResponseReasoningItem
    native = ResponseFunctionToolCall(type="function_call", name="wait", arguments="{}", call_id="sdk-call")
    reasoning = ResponseReasoningItem(type="reasoning", id="sdk-reason", summary=[])
    client = azure_client(lambda **_: SimpleNamespace(output=[reasoning, native], output_text="", status="completed"))
    result = client.complete_tools(instructions="system", input_items=[], tools=tool_definitions())
    assert str(result.single_action()) == "wait()"
    assert result.output[0] == {"type": "reasoning", "id": "sdk-reason", "summary": []}
    json.dumps(client.calls)


def test_native_evaluation_uses_tools_and_never_pools_legacy_results(tmp_path, monkeypatch):
    from motiverooms.evaluation import load_suite, summarize
    from scripts.evaluate import execute_plan
    config = dict(split="development", families=["private_handoffs"], seeds=[0],
                  conditions=["self_play"], both_seats=False, repeats=1,
                  communication_modes=["enabled", "disabled"], max_steps=2)
    monkeypatch.setattr("scripts.evaluate.AzureOpenAISettings.from_env",
                        lambda: AzureOpenAISettings("test", "https://example.com/openai/v1/", "test"))
    clients = []

    def create(**kwargs):
        return SimpleNamespace(output=list(response().output), output_text="", status="completed",
                               usage=dict(input_tokens=10, output_tokens=10, total_tokens=20))

    def factory(settings, budget):
        client = azure_client(create, budget)
        clients.append(client)
        return client

    records = execute_plan(config, make_plan(config, load_suite("rooms/diagnostics/manifest.json")),
                           tmp_path / "native", client_factory=factory)
    assert all(r["metadata"]["action_interface"] == "tools" for r in records)
    assert all(r["metrics"]["termination_reason"] == "max_steps" for r in records)
    assert all(len(c.calls) == 2 for c in clients)
    assert {t["name"] for t in clients[0].calls[0]["tools"]} > {t["name"] for t in clients[2].calls[0]["tools"]}
    for record in records:
        assert replay(record["path"]) == record["metrics"]
    legacy = deepcopy(records[0])
    legacy["metadata"].pop("action_interface")  # Historical logs predate this field.
    summary = summarize([records[0], legacy], tmp_path / "mixed")
    assert len(summary) == 2
    assert {s["action_interface"] for s in summary} == {"tools", "text"}
    conditions = json.loads((tmp_path / "mixed/condition_summary.json").read_text())
    assert len(conditions) == 2
