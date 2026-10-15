"""Azure OpenAI v1 Responses API (Foundry-compatible)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


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
        )

    def complete(self, *, instructions: str, user_input: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "instructions": instructions,
            "input": user_input,
            "max_output_tokens": self.settings.max_output_tokens,
        }
        if self.settings.temperature is not None:
            kwargs["temperature"] = self.settings.temperature
        response = self._client.responses.create(**kwargs)
        return _extract_output_text(response)


def mask_endpoint(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/…"
