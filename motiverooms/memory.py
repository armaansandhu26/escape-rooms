"""Bound native conversations at complete exchange boundaries."""
from __future__ import annotations

from copy import deepcopy


class ObservedActionMemory:
    """Bounded, verbatim own outcomes with attempt counts; never a state oracle."""

    def __init__(self, limit: int = 32):
        if type(limit) is not int or limit < 1:
            raise ValueError("Action memory limit must be a positive integer")
        self.limit = limit
        self._actions: dict[str, dict] = {}
        self._sequence = 0

    def record(self, action: str, *, valid: bool, result: str) -> None:
        self._sequence += 1
        previous = self._actions.pop(action, {})
        self._actions[action] = dict(action=action, attempts=previous.get("attempts", 0) + 1,
                                     last_valid=valid, last_result=result, last_own_turn=self._sequence)
        while len(self._actions) > self.limit:
            del self._actions[next(iter(self._actions))]

    def view(self) -> list[dict]:
        return deepcopy(list(self._actions.values()))


def validate_history_turns(turns: int | None) -> None:
    if turns is not None and (type(turns) is not int or turns < 1):
        raise ValueError("history_turns must be a positive integer or None")


def recent_exchanges(history: list[dict], turns: int | None) -> list[dict]:
    """Keep reasoning, calls and outputs together; call only after resolution."""
    if turns is None:
        return history
    starts = [index for index, item in enumerate(history) if item.get("role") == "user"]
    return history[starts[-turns]:] if len(starts) > turns else history
