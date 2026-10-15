import pytest

from motiverooms.actions import Action, ActionParseError, parse_action


def test_parser_uses_last_non_empty_line() -> None:
    output = "I should inspect the tag first.\n\ninspect(toolbox_tag)\n"

    assert parse_action(output) == Action("inspect", ("toolbox_tag",))


def test_spaced_item_names_coerce_to_identifiers() -> None:
    assert parse_action("use(brass key, exit)") == Action(
        "use", ("brass_key", "exit")
    )


def test_message_may_contain_commas() -> None:
    assert parse_action("say(The code is 4721, use it now)") == Action(
        "say", ("The code is 4721, use it now",)
    )


@pytest.mark.parametrize(
    "output",
    [
        "",
        "inspect()",
        "wait(extra)",
        "enter(not-a-code, cabinet)",
        "open(cabinet)",
    ],
)
def test_malformed_actions_are_rejected(output: str) -> None:
    with pytest.raises(ActionParseError):
        parse_action(output)
