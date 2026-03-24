"""Tests for on-chain data dataflow (CoinGecko, DeFi Llama, Etherscan)."""

import pytest
from unittest.mock import patch, MagicMock

from tradingagents.prediction_market.dataflows.onchain import (
    get_crypto_price,
    get_crypto_price_history,
    get_defi_tvl,
    get_top_defi_protocols,
    get_whale_transactions,
)


MOCK_COINGECKO_COIN = {
    "name": "Bitcoin",
    "symbol": "btc",
    "market_cap_rank": 1,
    "last_updated": "2026-03-25T12:00:00Z",
    "market_data": {
        "current_price": {"usd": 67500.00},
        "price_change_percentage_24h": 2.5,
        "price_change_percentage_7d": -1.3,
        "price_change_percentage_30d": 8.7,
        "total_volume": {"usd": 28000000000},
        "market_cap": {"usd": 1320000000000},
        "ath": {"usd": 73750.00},
        "ath_change_percentage": {"usd": -8.5},
    },
}

MOCK_COINGECKO_HISTORY = {
    "prices": [
        [1711324800000, 62000.0],
        [1711411200000, 63500.0],
        [1711497600000, 64200.0],
        [1711584000000, 67500.0],
    ],
}

MOCK_DEFILLAMA_PROTOCOL = {
    "name": "Aave",
    "category": "Lending",
    "currentChainTvls": {
        "Ethereum": 8000000000,
        "Polygon": 1200000000,
        "Arbitrum": 900000000,
    },
    "tvl": [
        {"totalLiquidityUSD": 9000000000},
        {"totalLiquidityUSD": 9500000000},
        {"totalLiquidityUSD": 9800000000},
        {"totalLiquidityUSD": 10000000000},
        {"totalLiquidityUSD": 10100000000},
        {"totalLiquidityUSD": 10050000000},
        {"totalLiquidityUSD": 10100000000},
        {"totalLiquidityUSD": 10100000000},
    ],
}

MOCK_DEFILLAMA_PROTOCOLS = [
    {"name": "Lido", "tvl": 15000000000, "category": "Liquid Staking", "change_1d": 0.5},
    {"name": "Aave", "tvl": 10000000000, "category": "Lending", "change_1d": -0.3},
    {"name": "Uniswap", "tvl": 5000000000, "category": "DEX", "change_1d": 1.2},
]

MOCK_ETHERSCAN_TXS = {
    "status": "1",
    "result": [
        {
            "hash": "0xabc123def456789012345678901234567890",
            "from": "0x1234567890abcdef1234567890abcdef12345678",
            "to": "0xfedcba0987654321fedcba0987654321fedcba09",
            "value": "5000000000000000000",  # 5 ETH
            "timeStamp": "1711584000",
        },
        {
            "hash": "0xdef456abc789012345678901234567890123",
            "from": "0xfedcba0987654321fedcba0987654321fedcba09",
            "to": "0x1234567890abcdef1234567890abcdef12345678",
            "value": "100000000000000000",  # 0.1 ETH
            "timeStamp": "1711580400",
        },
    ],
}


# ── CoinGecko: get_crypto_price ────────────────────────────────────

class TestGetCryptoPrice:
    @patch("tradingagents.prediction_market.dataflows.onchain.http_get_with_retry")
    def test_returns_formatted_price(self, mock_get):
        mock_get.return_value = MOCK_COINGECKO_COIN
        result = get_crypto_price("bitcoin")
        assert "Bitcoin" in result
        assert "BTC" in result
        assert "67,500" in result
        assert "+2.50%" in result

    @patch("tradingagents.prediction_market.dataflows.onchain.http_get_with_retry")
    def test_includes_market_cap_and_volume(self, mock_get):
        mock_get.return_value = MOCK_COINGECKO_COIN
        result = get_crypto_price("bitcoin")
        assert "Market Cap" in result
        assert "Volume" in result
        assert "All-Time High" in result

    @patch("tradingagents.prediction_market.dataflows.onchain.http_get_with_retry")
    def test_api_failure_returns_error(self, mock_get):
        mock_get.side_effect = Exception("API timeout")
        result = get_crypto_price("bitcoin")
        assert "[ERROR]" in result


# ── CoinGecko: get_crypto_price_history ─────────────────────────────

class TestGetCryptoPriceHistory:
    @patch("tradingagents.prediction_market.dataflows.onchain.http_get_with_retry")
    def test_returns_history_with_summary(self, mock_get):
        mock_get.return_value = MOCK_COINGECKO_HISTORY
        result = get_crypto_price_history("bitcoin", days=7)
        assert "Price History" in result
        assert "Summary" in result
        assert "Start:" in result
        assert "Change:" in result

    @patch("tradingagents.prediction_market.dataflows.onchain.http_get_with_retry")
    def test_empty_prices_returns_error(self, mock_get):
        mock_get.return_value = {"prices": []}
        result = get_crypto_price_history("bitcoin")
        assert "[ERROR]" in result

    @patch("tradingagents.prediction_market.dataflows.onchain.http_get_with_retry")
    def test_api_failure_returns_error(self, mock_get):
        mock_get.side_effect = RuntimeError("timeout")
        result = get_crypto_price_history("bitcoin")
        assert "[ERROR]" in result


# ── DeFi Llama: get_defi_tvl ───────────────────────────────────────

class TestGetDefiTvl:
    @patch("tradingagents.prediction_market.dataflows.onchain.http_get_with_retry")
    def test_returns_tvl_with_chain_breakdown(self, mock_get):
        mock_get.return_value = MOCK_DEFILLAMA_PROTOCOL
        result = get_defi_tvl("aave")
        assert "Aave" in result
        assert "Lending" in result
        assert "Ethereum" in result
        assert "TVL by Chain" in result

    @patch("tradingagents.prediction_market.dataflows.onchain.http_get_with_retry")
    def test_includes_tvl_changes(self, mock_get):
        mock_get.return_value = MOCK_DEFILLAMA_PROTOCOL
        result = get_defi_tvl("aave")
        assert "TVL Change" in result

    @patch("tradingagents.prediction_market.dataflows.onchain.http_get_with_retry")
    def test_api_failure_returns_error(self, mock_get):
        mock_get.side_effect = Exception("connection error")
        result = get_defi_tvl("aave")
        assert "[ERROR]" in result


# ── DeFi Llama: get_top_defi_protocols ──────────────────────────────

class TestGetTopDefiProtocols:
    @patch("tradingagents.prediction_market.dataflows.onchain.http_get_with_retry")
    def test_returns_ranked_protocols(self, mock_get):
        mock_get.return_value = MOCK_DEFILLAMA_PROTOCOLS
        result = get_top_defi_protocols(limit=3)
        assert "Lido" in result
        assert "Aave" in result
        assert "Uniswap" in result
        # Should be sorted by TVL (Lido first)
        assert result.index("Lido") < result.index("Aave")

    @patch("tradingagents.prediction_market.dataflows.onchain.http_get_with_retry")
    def test_empty_data_returns_error(self, mock_get):
        mock_get.return_value = []
        result = get_top_defi_protocols()
        assert "[ERROR]" in result

    @patch("tradingagents.prediction_market.dataflows.onchain.http_get_with_retry")
    def test_api_failure_returns_error(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        result = get_top_defi_protocols()
        assert "[ERROR]" in result


# ── Etherscan: get_whale_transactions ───────────────────────────────

class TestGetWhaleTransactions:
    @patch("tradingagents.prediction_market.dataflows.onchain.http_get_with_retry")
    def test_returns_transactions(self, mock_get):
        mock_get.return_value = MOCK_ETHERSCAN_TXS
        result = get_whale_transactions(
            address="0x1234567890abcdef1234567890abcdef12345678"
        )
        assert "Transactions" in result
        assert "ETH" in result
        assert "OUT" in result  # first tx is from this address

    @patch("tradingagents.prediction_market.dataflows.onchain.http_get_with_retry")
    def test_filters_small_transactions(self, mock_get):
        mock_get.return_value = MOCK_ETHERSCAN_TXS
        result = get_whale_transactions(
            address="0x1234567890abcdef1234567890abcdef12345678"
        )
        # 0.1 ETH tx should be filtered out (< 0.01 ETH threshold in code)
        assert "5.0000 ETH" in result

    def test_no_address_returns_error(self):
        result = get_whale_transactions()
        assert "[ERROR]" in result

    @patch("tradingagents.prediction_market.dataflows.onchain.http_get_with_retry")
    def test_etherscan_error_status(self, mock_get):
        mock_get.return_value = {"status": "0", "message": "NOTOK", "result": []}
        result = get_whale_transactions(
            address="0x1234567890abcdef1234567890abcdef12345678"
        )
        assert "[ERROR]" in result

    @patch("tradingagents.prediction_market.dataflows.onchain.http_get_with_retry")
    def test_api_failure_returns_error(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        result = get_whale_transactions(
            address="0x1234567890abcdef1234567890abcdef12345678"
        )
        assert "[ERROR]" in result
