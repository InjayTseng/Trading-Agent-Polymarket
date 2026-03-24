"""Tests for shared HTTP utilities — retry, backoff, rate limit handling."""

import pytest
from unittest.mock import patch, MagicMock

from tradingagents.prediction_market.dataflows.http_utils import http_get_with_retry


class TestHttpGetWithRetry:
    @patch("tradingagents.prediction_market.dataflows.http_utils.requests.get")
    def test_successful_request(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = http_get_with_retry("https://example.com/api")
        assert result == {"data": "ok"}
        mock_get.assert_called_once()

    @patch("tradingagents.prediction_market.dataflows.http_utils.requests.get")
    def test_passes_params_and_timeout(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        http_get_with_retry("https://example.com", params={"q": "test"}, timeout=15)
        mock_get.assert_called_once_with(
            "https://example.com", params={"q": "test"}, timeout=15
        )

    @patch("tradingagents.prediction_market.dataflows.http_utils.time.sleep")
    @patch("tradingagents.prediction_market.dataflows.http_utils.requests.get")
    def test_429_rate_limit_retries(self, mock_get, mock_sleep):
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "2"}

        success = MagicMock()
        success.status_code = 200
        success.json.return_value = {"ok": True}
        success.raise_for_status = MagicMock()

        mock_get.side_effect = [rate_limited, success]

        result = http_get_with_retry("https://example.com", retries=3)
        assert result == {"ok": True}
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(2)

    @patch("tradingagents.prediction_market.dataflows.http_utils.time.sleep")
    @patch("tradingagents.prediction_market.dataflows.http_utils.requests.get")
    def test_connection_error_retries_with_backoff(self, mock_get, mock_sleep):
        from requests.exceptions import ConnectionError

        success = MagicMock()
        success.status_code = 200
        success.json.return_value = {"ok": True}
        success.raise_for_status = MagicMock()

        mock_get.side_effect = [ConnectionError("fail"), success]

        result = http_get_with_retry("https://example.com", retries=3, backoff_base=2.0)
        assert result == {"ok": True}
        assert mock_get.call_count == 2
        # First retry: backoff_base ** 0 = 1.0
        mock_sleep.assert_called_once_with(1.0)

    @patch("tradingagents.prediction_market.dataflows.http_utils.time.sleep")
    @patch("tradingagents.prediction_market.dataflows.http_utils.requests.get")
    def test_max_retries_exhausted_raises(self, mock_get, mock_sleep):
        from requests.exceptions import Timeout

        mock_get.side_effect = Timeout("timeout")

        with pytest.raises(Timeout):
            http_get_with_retry("https://example.com", retries=3)
        assert mock_get.call_count == 3

    @patch("tradingagents.prediction_market.dataflows.http_utils.time.sleep")
    @patch("tradingagents.prediction_market.dataflows.http_utils.requests.get")
    def test_429_exhausts_all_retries(self, mock_get, mock_sleep):
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {}

        mock_get.return_value = rate_limited

        with pytest.raises(RuntimeError, match="Failed after"):
            http_get_with_retry("https://example.com", retries=2)
        assert mock_get.call_count == 2

    @patch("tradingagents.prediction_market.dataflows.http_utils.requests.get")
    def test_http_error_raised_immediately(self, mock_get):
        from requests.exceptions import HTTPError

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = HTTPError("500 Server Error")
        mock_get.return_value = mock_resp

        with pytest.raises(HTTPError):
            http_get_with_retry("https://example.com", retries=1)
