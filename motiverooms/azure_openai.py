"""Azure OpenAI v1 Responses API (Foundry-compatible)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

from motiverooms.telemetry import RequestBudget
from motiverooms.tool_calls import ToolResponse


class AzureOpenAIConfigError(ValueError):
    """Missing or invalid Azure OpenAI settings."""


def _normalize_base_url(raw: str) -> str:
    text = raw.strip().rstrip("/")
    if text.endswith("/responses"):
        text = text[: -len("/responses")]
    if not text.endswith("/openai/v1"):
        if text.endswith("/openai"):
            text = f"{text}/v1"
        else:
            text = f"{text}/openai/v1"
    return f"{text}/"


@dataclass(frozen=True)
class AzureOpenAISettings:
    api_key: str
    base_url: str
    model: str
    max_output_tokens: int = 300
    temperature: float | None = 0.2
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> AzureOpenAISettings:
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
        if not api_key:
            raise AzureOpenAIConfigError(
                "Set AZURE_OPENAI_API_KEY (Foundry project key or Azure OpenAI key)."
            )

        base = os.environ.get(
            "AZURE_OPENAI_BASE_URL",
            "https://armaan-foundry.services.ai.azure.com/openai/v1/",
        ).strip()
        model = os.environ.get("AZURE_OPENAI_MODEL", "gpt-5.4-mini").strip()
        max_tokens = int(os.environ.get("AZURE_OPENAI_MAX_OUTPUT_TOKENS", "300"))
        temp_raw = os.environ.get("AZURE_OPENAI_TEMPERATURE", "0.2").strip()
        temperature = None if temp_raw.lower() in {"", "default", "none"} else float(temp_raw)
        timeout = float(os.environ.get("AZURE_OPENAI_TIMEOUT", "120"))

        return cls(
            api_key=api_key,
            base_url=_normalize_base_url(base),
            model=model,
            max_output_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout,
        )

    @classmethod
    def preview_from_env(cls) -> AzureOpenAISettings:
        """Non-secret settings for --dry-run (api_key may be placeholder)."""
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip() or "(unset)"
        base = os.environ.get(
            "AZURE_OPENAI_BASE_URL",
            "https://armaan-foundry.services.ai.azure.com/openai/v1/",
        ).strip()
        model = os.environ.get("AZURE_OPENAI_MODEL", "gpt-5.4-mini").strip()
        max_tokens = int(os.environ.get("AZURE_OPENAI_MAX_OUTPUT_TOKENS", "300"))
        temp_raw = os.environ.get("AZURE_OPENAI_TEMPERATURE", "0.2").strip()
        temperature = None if temp_raw.lower() in {"", "default", "none"} else float(temp_raw)
        timeout = float(os.environ.get("AZURE_OPENAI_TIMEOUT", "120"))
        return cls(
            api_key=api_key,
            base_url=_normalize_base_url(base),
            model=model,
            max_output_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout,
        )


def _extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = getattr(response, "output", None)
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for part in content:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str):
                        chunks.append(part_text)
            elif isinstance(item, dict):
                for part in item.get("content", []):
                    if isinstance(part, dict) and part.get("text"):
                        chunks.append(str(part["text"]))
        if chunks:
            return "\n".join(chunks).strip()

    raise RuntimeError("Azure response contained no text output.")


@dataclass
class AzureResponsesClient:
    settings: AzureOpenAISettings
    budget: RequestBudget | None = None
    calls: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError(
                "Install Azure support: pip install -e '.[azure]'"
            ) from error
        self._client = OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout=self.settings.timeout_seconds,
            max_retries=0,  # Every provider request must be visible in accounting.
        )

    def complete(self, *, instructions: str, user_input: str) -> str:
        """Legacy text completion, retained for historical experiment protocols."""
        record = self._request(instructions=instructions, user_input=user_input)
        return record["output"]

    def complete_tools(self, *, instructions: str, input_items: list[dict],
                       tools: list[dict]) -> ToolResponse:
        record = self._request(
            instructions=instructions, user_input=input_items,
            tools=tools, tool_choice="required", parallel_tool_calls=False,
        )
        return ToolResponse(tuple(record["output_items"]), record["output"], record["status"])

    def _request(self, *, instructions: str, user_input: str | list[dict], **options: Any) -> dict[str, Any]:
        """One accounted provider request; both interfaces share transport and logging."""
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "instructions": instructions,
            "input": user_input,
            "max_output_tokens": self.settings.max_output_tokens,
            **options,
        }
        if self.settings.temperature is not None:
            kwargs["temperature"] = self.settings.temperature
        # Tool schemas and the full private transcript also consume input tokens.
        budget_input = user_input if isinstance(user_input, str) else json.dumps(user_input)
        if options:
            budget_input += json.dumps(options)
        reservation = self.budget.reserve(instructions, budget_input, self.settings.max_output_tokens) if self.budget else None
        record = dict(model=self.settings.model, instructions=instructions, input=user_input,
                      max_output_tokens=self.settings.max_output_tokens,
                      temperature=self.settings.temperature, usage=None, output=None, error=None)
        record.update(options)
        started = perf_counter()
        try:
            response = self._client.responses.create(**kwargs)
            raw_usage = getattr(response, "usage", None)
            if raw_usage is not None:
                usage = raw_usage if isinstance(raw_usage, dict) else {
                    key: getattr(raw_usage, key, 0) for key in ("input_tokens", "output_tokens", "total_tokens")
                }
                record["usage"] = {key: int(usage.get(key, 0) or 0) for key in
                                   ("input_tokens", "output_tokens", "total_tokens")}
                if not record["usage"]["total_tokens"]:
                    record["usage"]["total_tokens"] = record["usage"]["input_tokens"] + record["usage"]["output_tokens"]
            record["response_id"] = getattr(response, "id", None)
            record["status"] = getattr(response, "status", None)
            record["raw_text"] = getattr(response, "output_text", None)
            record["output_items"] = [
                item if isinstance(item, dict) else item.model_dump(mode="json", exclude_none=True)
                for item in (getattr(response, "output", None) or [])
            ]
            try:
                record["output"] = _extract_output_text(response)
            except RuntimeError:
                # Native calls commonly have no final text. In text mode this
                # is a model-output failure, charged as an invalid action.
                record["output"] = ""
                if not options:
                    record["output_error"] = "empty_text"
            return record
        except Exception as error:
            record["error"] = type(error).__name__
            raise
        finally:
            record["latency_seconds"] = perf_counter() - started
            record["estimated_cost_usd"] = (
                self.budget.estimate_cost(record["usage"]["input_tokens"], record["usage"]["output_tokens"])
                if self.budget and record["usage"] else None
            )
            self.calls.append(record)
            if self.budget and reservation:
                self.budget.settle(reservation, record["usage"])


def mask_endpoint(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/…"
