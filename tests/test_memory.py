import pytest

from motiverooms.memory import recent_exchanges, validate_history_turns


def test_trimming_preserves_complete_calls_results_and_reasoning_items():
    turns = [[dict(role="user", content=str(i)), dict(type="reasoning", id=f"reason-{i}"),
              dict(type="function_call", call_id=f"call-{i}"),
              dict(type="function_call_output", call_id=f"call-{i}", output="ok")]
             for i in range(6)]
    history = [item for turn in turns for item in turn]
    assert recent_exchanges(history, None) == history
    assert recent_exchanges(history, 2) == turns[-2] + turns[-1]
    assert recent_exchanges(history, 2)[0]["role"] == "user"
    assert recent_exchanges(history, 20) == history


@pytest.mark.parametrize("value", [0, -1, True, "3", 1.5])
def test_bad_memory_limits_fail_before_provider_requests(value):
    with pytest.raises(ValueError):
        validate_history_turns(value)
