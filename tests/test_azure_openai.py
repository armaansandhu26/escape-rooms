import pytest

from motiverooms.azure_openai import (
    AzureOpenAIConfigError,
    AzureOpenAISettings,
    _normalize_base_url,
)


def test_normalize_responses_url_to_v1_base() -> None:
    assert (
        _normalize_base_url(
            "https://armaan-foundry.services.ai.azure.com/openai/v1/responses"
        )
        == "https://armaan-foundry.services.ai.azure.com/openai/v1/"
    )


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv(
        "AZURE_OPENAI_BASE_URL",
        "https://example.services.ai.azure.com/openai/v1/",
    )
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-5.4-mini")
    settings = AzureOpenAISettings.from_env()
    assert settings.model == "gpt-5.4-mini"
    assert settings.api_key == "test-key"


def test_settings_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    with pytest.raises(AzureOpenAIConfigError):
        AzureOpenAISettings.from_env()
