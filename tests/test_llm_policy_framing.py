from motiverooms.azure_openai import AzureOpenAISettings
from motiverooms.llm_policy import LLMAgentPolicy, _shared_escape_goal


class _StubClient:
    settings = AzureOpenAISettings(
        api_key="x",
        base_url="https://example.com/openai/v1/",
        model="test",
    )

    def complete(self, *, instructions: str, user_input: str) -> str:
        del user_input
        return instructions


def test_competitive_prompt_does_not_say_cooperate() -> None:
    policy = LLMAgentPolicy(
        agent="A", client=_StubClient(), framing="competitive", mode="competitive",
    )
    prompt = policy.system_prompt()
    assert "points minus theirs" in prompt
    assert "Withhold codes" not in prompt
    assert "Share codes" not in prompt
    assert "Cooperate with your partner" not in prompt


def test_shared_escape_goal_varies_by_mechanics() -> None:
    assert "every agent" in _shared_escape_goal("together")
    assert "first" in _shared_escape_goal("race").lower()


def test_coaching_is_an_explicit_condition() -> None:
    policy = LLMAgentPolicy(
        agent="B", client=_StubClient(), framing="competitive", coaching=True,
        shared_room=True, shared_escape="together",
    )
    prompt = policy.system_prompt()
    assert "Withhold codes" in prompt
    assert "every agent escape" in prompt
    assert "first agent" not in prompt


def test_split_room_prompt_does_not_claim_shared_exit() -> None:
    prompt = LLMAgentPolicy(agent="A", client=_StubClient()).system_prompt()
    assert "opening your door" in prompt
    assert "shared exit" not in prompt
