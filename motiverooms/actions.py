"""Parsing for the deliberately small MotiveRooms action language."""

from __future__ import annotations

from dataclasses import dataclass
import re

ACTION_RE = re.compile(
    r"^(inspect|take|enter|use|give|say|write_note|read_notes|wait)\((.*)\)\s*$"
)
IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

ACTION_ARITY = {
    "inspect": 1,
    "take": 1,
    "enter": 2,
    "use": 2,
    "give": 1,
    "say": 1,
    "write_note": 1,
    "read_notes": 0,
    "wait": 0,
}


class ActionParseError(ValueError):
    """Raised when model output does not end with one valid action."""


@dataclass(frozen=True)
class Action:
    verb: str
    args: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.verb}({', '.join(self.args)})"


def parse_action(output: str) -> Action:
    """Parse the last non-empty line of an agent response.

    Message actions keep their complete text, including commas.  All other
    actions have fixed arity and identifier-like node/item arguments.
    """

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise ActionParseError("Response is empty")

    action_line = lines[-1]
    match = ACTION_RE.fullmatch(action_line)
    if match is None:
        raise ActionParseError("Last line must contain exactly one valid action")

    verb, raw_arguments = match.groups()
    if verb in {"say", "write_note"}:
        arguments = (raw_arguments.strip(),)
    elif not raw_arguments.strip():
        arguments = ()
    else:
        arguments = tuple(part.strip() for part in raw_arguments.split(","))

    expected = ACTION_ARITY[verb]
    if len(arguments) != expected or any(not argument for argument in arguments):
        raise ActionParseError(f"{verb} expects {expected} argument(s)")

    if verb not in {"say", "write_note", "read_notes", "wait"}:
        identifier_arguments = arguments[1:] if verb == "enter" else arguments
        try:
            coerced = tuple(_coerce_identifier(arg) for arg in identifier_arguments)
        except ActionParseError:
            raise ActionParseError("Node and item ids must be identifiers") from None
        if verb == "enter":
            arguments = (arguments[0], coerced[0])
        else:
            arguments = coerced

    if verb == "enter" and not arguments[0].isdigit():
        raise ActionParseError("enter expects a numeric code")

    return Action(verb=verb, args=arguments)


def _coerce_identifier(raw: str) -> str:
    text = raw.strip()
    if IDENTIFIER_RE.fullmatch(text):
        return text
    collapsed = re.sub(r"[\s-]+", "_", text)
    if IDENTIFIER_RE.fullmatch(collapsed):
        return collapsed
    raise ActionParseError("Node and item ids must be identifiers")
