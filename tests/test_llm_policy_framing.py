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
    policy = LLMAgentPolicy(agent="A", client=_StubClient(), framing="competitive")
    prompt = policy.system_prompt()
    assert "points minus theirs" in prompt
    assert "Withhold codes" in prompt
    assert "Cooperate with your partner" not in prompt


def test_shared_escape_goal_varies_by_framing() -> None:
    assert "Cooperate" in _shared_escape_goal("collaborative")
    assert "first" in _shared_escape_goal("competitive").lower()
