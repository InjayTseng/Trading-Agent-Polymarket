"""Tests for LLM client factory and provider routing."""

import warnings
from unittest.mock import patch, MagicMock

import pytest

from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.validators import validate_model, VALID_MODELS


# ── Provider routing ────────────────────────────────────────────────

class TestProviderRouting:
    @patch("tradingagents.llm_clients.openai_client.ChatOpenAI.__init__", return_value=None)
    def test_openai_provider(self, mock_init):
        client = create_llm_client("openai", "gpt-5-mini")
        llm = client.get_llm()
        mock_init.assert_called_once()

    @patch("tradingagents.llm_clients.openai_client.ChatOpenAI.__init__", return_value=None)
    def test_xai_provider_sets_base_url(self, mock_init):
        with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}):
            client = create_llm_client("xai", "grok-4-fast-reasoning")
            llm = client.get_llm()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["base_url"] == "https://api.x.ai/v1"

    @patch("tradingagents.llm_clients.openai_client.ChatOpenAI.__init__", return_value=None)
    def test_ollama_provider_default_url(self, mock_init):
        client = create_llm_client("ollama", "llama3")
        llm = client.get_llm()
        call_kwargs = mock_init.call_args[1]
        assert "localhost:11434" in call_kwargs["base_url"]

    @patch("tradingagents.llm_clients.openai_client.ChatOpenAI.__init__", return_value=None)
    def test_openrouter_provider(self, mock_init):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            client = create_llm_client("openrouter", "some-model")
            llm = client.get_llm()
        call_kwargs = mock_init.call_args[1]
        assert "openrouter.ai" in call_kwargs["base_url"]

    @patch("tradingagents.llm_clients.anthropic_client.ChatAnthropic", autospec=True)
    def test_anthropic_provider(self, mock_cls):
        mock_cls.return_value = MagicMock()
        client = create_llm_client("anthropic", "claude-sonnet-4-6")
        llm = client.get_llm()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"

    @patch("tradingagents.llm_clients.google_client.ChatGoogleGenerativeAI.__init__", return_value=None)
    def test_google_provider(self, mock_init):
        client = create_llm_client("google", "gemini-2.5-flash")
        llm = client.get_llm()
        mock_init.assert_called_once()

    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            create_llm_client("invalid_provider", "model")

    def test_case_insensitive_provider(self):
        """Provider name should be case insensitive."""
        with patch("tradingagents.llm_clients.anthropic_client.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            client = create_llm_client("Anthropic", "claude-sonnet-4-6")
            assert client is not None


# ── API key mapping ─────────────────────────────────────────────────

class TestApiKeyMapping:
    @patch("tradingagents.llm_clients.google_client.ChatGoogleGenerativeAI.__init__", return_value=None)
    def test_google_api_key_mapped(self, mock_init):
        client = create_llm_client("google", "gemini-2.5-flash", api_key="test-key")
        llm = client.get_llm()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["google_api_key"] == "test-key"

    @patch("tradingagents.llm_clients.google_client.ChatGoogleGenerativeAI.__init__", return_value=None)
    def test_google_native_key_not_overridden(self, mock_init):
        client = create_llm_client("google", "gemini-2.5-flash", google_api_key="native-key", api_key="generic-key")
        llm = client.get_llm()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["google_api_key"] == "native-key"


# ── base_url passthrough ────────────────────────────────────────────

class TestBaseUrlPassthrough:
    @patch("tradingagents.llm_clients.anthropic_client.ChatAnthropic", autospec=True)
    def test_anthropic_base_url_passed(self, mock_cls):
        mock_cls.return_value = MagicMock()
        client = create_llm_client("anthropic", "claude-sonnet-4-6", base_url="https://custom.proxy/v1")
        llm = client.get_llm()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["anthropic_api_url"] == "https://custom.proxy/v1"

    @patch("tradingagents.llm_clients.anthropic_client.ChatAnthropic", autospec=True)
    def test_anthropic_no_base_url_omits_param(self, mock_cls):
        mock_cls.return_value = MagicMock()
        client = create_llm_client("anthropic", "claude-sonnet-4-6")
        llm = client.get_llm()
        call_kwargs = mock_cls.call_args[1]
        assert "anthropic_api_url" not in call_kwargs


# ── Model validation ────────────────────────────────────────────────

class TestModelValidation:
    def test_valid_openai_model(self):
        assert validate_model("openai", "gpt-5-mini") is True

    def test_invalid_openai_model(self):
        assert validate_model("openai", "nonexistent-model") is False

    def test_valid_anthropic_model(self):
        assert validate_model("anthropic", "claude-opus-4-6") is True

    def test_ollama_accepts_any_model(self):
        assert validate_model("ollama", "anything-goes") is True

    def test_openrouter_accepts_any_model(self):
        assert validate_model("openrouter", "custom/model") is True

    def test_unknown_provider_warns_and_accepts(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_model("unknown_provider", "model")
            assert result is True
            assert len(w) == 1
            assert "Unknown provider" in str(w[0].message)

    def test_factory_warns_on_invalid_model(self):
        """create_llm_client should warn but not error for unknown models."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("tradingagents.llm_clients.anthropic_client.ChatAnthropic") as mock_cls:
                mock_cls.return_value = MagicMock()
                client = create_llm_client("anthropic", "definitely-not-a-model")
            assert any("may not be valid" in str(warning.message) for warning in w)
