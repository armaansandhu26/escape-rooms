"""Shared request accounting and conservative pre-call budget reservations."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from math import isfinite


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class RequestBudget:
    max_calls: int = 100
    max_tokens: int = 250_000
    max_cost_usd: float | None = None
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    calls: int = 0
    tokens: int = 0
    reserved_tokens: int = 0
    cost_usd: float = 0.0
    reserved_cost: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def __post_init__(self):
        if self.max_calls < 1 or self.max_tokens < 1:
            raise ValueError("Call and token budgets must be positive")
        rates = (self.input_usd_per_million, self.output_usd_per_million)
        if any(r is not None and (not isfinite(r) or r < 0) for r in rates):
            raise ValueError("Prices must be nonnegative")
        if self.max_cost_usd is not None:
            if not isfinite(self.max_cost_usd) or self.max_cost_usd <= 0 or any(r is None for r in rates):
                raise ValueError("A dollar budget requires a positive cap and both token prices")

    def estimate_cost(self, input_tokens, output_tokens):
        if self.input_usd_per_million is None or self.output_usd_per_million is None:
            return None
        return (input_tokens * self.input_usd_per_million +
                output_tokens * self.output_usd_per_million) / 1_000_000

    def reserve(self, instructions, user_input, max_output):
        # Byte count plus protocol allowance is conservative for typical text BPE
        # deployments, not a provider-guaranteed tokenizer count.
        input_bound = len((instructions + user_input).encode("utf-8")) + 256
        tokens = input_bound + max_output
        cost = self.estimate_cost(input_bound, max_output) or 0.0
        with self._lock:
            if self.calls >= self.max_calls:
                raise BudgetExceeded("API call budget reached")
            if self.tokens + self.reserved_tokens + tokens > self.max_tokens:
                raise BudgetExceeded("Token budget would be exceeded by next request reservation")
            if self.max_cost_usd is not None and self.cost_usd + self.reserved_cost + cost > self.max_cost_usd:
                raise BudgetExceeded("Estimated dollar budget would be exceeded by next request")
            self.calls += 1
            self.reserved_tokens += tokens
            self.reserved_cost += cost
        return tokens, cost

    def settle(self, reservation, usage):
        reserved_tokens, reserved_cost = reservation
        # Unknown usage (including a timed-out request) retains its reservation
        # as conservative spending; it must not be treated as free.
        tokens = usage["total_tokens"] if usage else reserved_tokens
        cost = self.estimate_cost(usage["input_tokens"], usage["output_tokens"]) if usage else reserved_cost
        with self._lock:
            self.reserved_tokens -= reserved_tokens
            self.reserved_cost -= reserved_cost
            self.tokens += tokens
            self.cost_usd += cost or 0.0
