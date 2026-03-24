"""Tests for news.py dataflow — NewsAPI, yfinance fallback, locale params."""

import pytest
from unittest.mock import patch, MagicMock

from tradingagents.prediction_market.dataflows.news import (
    get_pm_news,
    get_pm_global_news,
    _newsapi_search,
    _yfinance_news,
)


MOCK_NEWSAPI_RESPONSE = {
    "status": "ok",
    "totalResults": 3,
    "articles": [
        {
            "title": "Bitcoin surges past $70,000",
            "source": {"name": "CoinDesk"},
            "publishedAt": "2026-03-24T10:00:00Z",
            "description": "Bitcoin has broken through the $70,000 barrier amid renewed institutional interest.",
        },
        {
            "title": "Federal Reserve holds rates steady",
            "source": {"name": "Reuters"},
            "publishedAt": "2026-03-23T15:00:00Z",
            "description": "The Federal Reserve maintained interest rates, citing mixed economic signals.",
        },
    ],
}

MOCK_NEWSAPI_HEADLINES = {
    "status": "ok",
    "totalResults": 2,
    "articles": [
        {
            "title": "Global markets rally on trade deal hopes",
            "source": {"name": "Bloomberg"},
            "publishedAt": "2026-03-25T08:00:00Z",
        },
    ],
}


# ── _newsapi_search ─────────────────────────────────────────────────

class TestNewsApiSearch:
    @patch("tradingagents.prediction_market.dataflows.news.http_get_with_retry")
    @patch.dict("os.environ", {"NEWSAPI_KEY": "test-key"})
    def test_returns_formatted_results(self, mock_get):
        mock_get.return_value = MOCK_NEWSAPI_RESPONSE
        result = _newsapi_search("bitcoin")
        assert result is not None
        assert "Bitcoin surges" in result
        assert "CoinDesk" in result
        assert "Total results: 3" in result

    @patch("tradingagents.prediction_market.dataflows.news.http_get_with_retry")
    @patch.dict("os.environ", {"NEWSAPI_KEY": "test-key"})
    def test_language_param_passed(self, mock_get):
        mock_get.return_value = MOCK_NEWSAPI_RESPONSE
        _newsapi_search("bitcoin", language="de")
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["params"]["language"] == "de"

    @patch.dict("os.environ", {}, clear=True)
    def test_no_api_key_returns_none(self):
        result = _newsapi_search("bitcoin")
        assert result is None

    @patch("tradingagents.prediction_market.dataflows.news.http_get_with_retry")
    @patch.dict("os.environ", {"NEWSAPI_KEY": "test-key"})
    def test_empty_articles_returns_none(self, mock_get):
        mock_get.return_value = {"articles": []}
        result = _newsapi_search("obscure query")
        assert result is None

    @patch("tradingagents.prediction_market.dataflows.news.http_get_with_retry")
    @patch.dict("os.environ", {"NEWSAPI_KEY": "test-key"})
    def test_api_failure_returns_none(self, mock_get):
        mock_get.side_effect = Exception("API timeout")
        result = _newsapi_search("bitcoin")
        assert result is None


# ── get_pm_news (fallback chain) ────────────────────────────────────

class TestGetPmNews:
    @patch("tradingagents.prediction_market.dataflows.news.http_get_with_retry")
    @patch.dict("os.environ", {"NEWSAPI_KEY": "test-key"})
    def test_uses_newsapi_first(self, mock_get):
        mock_get.return_value = MOCK_NEWSAPI_RESPONSE
        result = get_pm_news("bitcoin")
        assert "via NewsAPI" in result

    @patch("tradingagents.prediction_market.dataflows.news._yfinance_news")
    @patch("tradingagents.prediction_market.dataflows.news._newsapi_search")
    def test_falls_back_to_yfinance(self, mock_newsapi, mock_yf):
        mock_newsapi.return_value = None
        mock_yf.return_value = "Financial news for: 'BTC' (via Yahoo Finance)\n"
        result = get_pm_news("BTC")
        assert "Yahoo Finance" in result

    @patch("tradingagents.prediction_market.dataflows.news._yfinance_news")
    @patch("tradingagents.prediction_market.dataflows.news._newsapi_search")
    def test_returns_fallback_message_when_all_fail(self, mock_newsapi, mock_yf):
        mock_newsapi.return_value = None
        mock_yf.return_value = None
        result = get_pm_news("obscure topic")
        assert "No news found" in result
        assert "NEWSAPI_KEY" in result

    @patch("tradingagents.prediction_market.dataflows.news.http_get_with_retry")
    @patch.dict("os.environ", {"NEWSAPI_KEY": "test-key"})
    def test_language_param_forwarded(self, mock_get):
        mock_get.return_value = MOCK_NEWSAPI_RESPONSE
        get_pm_news("bitcoin", language="fr")
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["params"]["language"] == "fr"


# ── get_pm_global_news ──────────────────────────────────────────────

class TestGetPmGlobalNews:
    @patch("tradingagents.prediction_market.dataflows.news.http_get_with_retry")
    @patch.dict("os.environ", {"NEWSAPI_KEY": "test-key"})
    def test_returns_headlines(self, mock_get):
        mock_get.return_value = MOCK_NEWSAPI_HEADLINES
        result = get_pm_global_news()
        assert "Headlines" in result
        assert "Bloomberg" in result

    @patch("tradingagents.prediction_market.dataflows.news.http_get_with_retry")
    @patch.dict("os.environ", {"NEWSAPI_KEY": "test-key"})
    def test_country_param_passed(self, mock_get):
        mock_get.return_value = MOCK_NEWSAPI_HEADLINES
        get_pm_global_news(country="gb")
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["params"]["country"] == "gb"

    @patch("tradingagents.prediction_market.dataflows.news._yfinance_news")
    @patch.dict("os.environ", {}, clear=True)
    def test_falls_back_without_api_key(self, mock_yf):
        mock_yf.return_value = "Financial news for: 'world news' (via Yahoo Finance)\n"
        result = get_pm_global_news()
        assert "Yahoo Finance" in result

    @patch("tradingagents.prediction_market.dataflows.news._yfinance_news")
    @patch.dict("os.environ", {}, clear=True)
    def test_returns_fallback_message_when_all_fail(self, mock_yf):
        mock_yf.return_value = None
        result = get_pm_global_news()
        assert "No global news available" in result
