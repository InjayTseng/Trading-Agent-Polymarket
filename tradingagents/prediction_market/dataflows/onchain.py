"""On-chain data fetching for crypto-related prediction markets.

Uses free public APIs (no authentication required):
- CoinGecko: Token prices, volume, market cap
- DeFi Llama: TVL (Total Value Locked) data for DeFi protocols
- Etherscan: Whale transaction monitoring (requires free API key for higher limits)

Set ETHERSCAN_API_KEY in .env for whale transaction monitoring.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

from .http_utils import http_get_with_retry

logger = logging.getLogger(__name__)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
DEFILLAMA_BASE = "https://api.llama.fi"


def get_crypto_price(coin_id: str, currency: str = "usd") -> str:
    """Get current price, volume, and market data for a cryptocurrency.

    Args:
        coin_id: CoinGecko coin ID (e.g., "bitcoin", "ethereum", "solana")
        currency: Fiat currency for pricing (default: "usd")

    Returns:
        Formatted string with price, volume, market cap, and 24h change
    """
    try:
        data = http_get_with_retry(
            f"{COINGECKO_BASE}/coins/{coin_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "community_data": "false",
                "developer_data": "false",
            },
            timeout=15,
            retries=2,
        )
    except Exception as e:
        logger.warning("CoinGecko request failed for %s: %s", coin_id, e)
        return f"[ERROR] Failed to fetch price data for '{coin_id}': {e}"

    market = data.get("market_data", {})
    cur = currency.lower()

    price = market.get("current_price", {}).get(cur)
    change_24h = market.get("price_change_percentage_24h")
    change_7d = market.get("price_change_percentage_7d")
    change_30d = market.get("price_change_percentage_30d")
    volume = market.get("total_volume", {}).get(cur)
    mcap = market.get("market_cap", {}).get(cur)
    ath = market.get("ath", {}).get(cur)
    ath_change = market.get("ath_change_percentage", {}).get(cur)

    lines = [
        f"Crypto: {data.get('name', coin_id)} ({data.get('symbol', '').upper()})",
        f"Price: ${price:,.2f}" if price else "Price: N/A",
        "",
        "Price Changes:",
        f"  24h: {change_24h:+.2f}%" if change_24h is not None else "  24h: N/A",
        f"  7d:  {change_7d:+.2f}%" if change_7d is not None else "  7d: N/A",
        f"  30d: {change_30d:+.2f}%" if change_30d is not None else "  30d: N/A",
        "",
        f"24h Volume: ${volume:,.0f}" if volume else "24h Volume: N/A",
        f"Market Cap: ${mcap:,.0f}" if mcap else "Market Cap: N/A",
        f"All-Time High: ${ath:,.2f} ({ath_change:+.1f}% from ATH)" if ath else "All-Time High: N/A",
        "",
        f"Market Cap Rank: #{data.get('market_cap_rank', 'N/A')}",
        f"Last Updated: {data.get('last_updated', 'N/A')}",
    ]

    return "\n".join(lines)


def get_crypto_price_history(coin_id: str, days: int = 30, currency: str = "usd") -> str:
    """Get historical price data for a cryptocurrency.

    Args:
        coin_id: CoinGecko coin ID
        days: Number of days of history (1, 7, 14, 30, 90, 180, 365, max)
        currency: Fiat currency

    Returns:
        Formatted string with price history and summary stats
    """
    try:
        data = http_get_with_retry(
            f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
            params={"vs_currency": currency, "days": days},
            timeout=15,
            retries=2,
        )
    except Exception as e:
        logger.warning("CoinGecko history request failed for %s: %s", coin_id, e)
        return f"[ERROR] Failed to fetch price history for '{coin_id}': {e}"

    prices = data.get("prices", [])
    if not prices:
        return f"[ERROR] No price history available for '{coin_id}'"

    # Sample at most 20 data points for readability
    step = max(1, len(prices) // 20)
    sampled = prices[::step]

    lines = [
        f"Price History: {coin_id} (last {days} days)",
        f"Data points: {len(prices)} (showing {len(sampled)} samples)",
        "",
        "Date | Price",
        "--- | ---",
    ]

    for ts, price in sampled:
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        lines.append(f"{dt} | ${price:,.2f}")

    # Summary
    all_prices = [p for _, p in prices]
    lines.extend([
        "",
        "Summary:",
        f"  Start: ${all_prices[0]:,.2f}",
        f"  End:   ${all_prices[-1]:,.2f}",
        f"  Min:   ${min(all_prices):,.2f}",
        f"  Max:   ${max(all_prices):,.2f}",
        f"  Change: {((all_prices[-1] / all_prices[0]) - 1) * 100:+.2f}%",
    ])

    return "\n".join(lines)


def get_defi_tvl(protocol: str) -> str:
    """Get Total Value Locked (TVL) data for a DeFi protocol.

    Args:
        protocol: DeFi Llama protocol slug (e.g., "aave", "uniswap", "lido")

    Returns:
        Formatted string with current TVL and recent changes
    """
    try:
        data = http_get_with_retry(
            f"{DEFILLAMA_BASE}/protocol/{protocol}",
            timeout=15,
            retries=2,
        )
    except Exception as e:
        logger.warning("DeFi Llama request failed for %s: %s", protocol, e)
        return f"[ERROR] Failed to fetch TVL data for '{protocol}': {e}"

    name = data.get("name", protocol)
    current_tvl = data.get("currentChainTvls", {})
    total_tvl = sum(v for k, v in current_tvl.items() if not k.endswith("-borrowed") and not k.endswith("-staking"))

    lines = [
        f"DeFi Protocol: {name}",
        f"Category: {data.get('category', 'N/A')}",
        f"Total TVL: ${total_tvl:,.0f}" if total_tvl else "Total TVL: N/A",
        "",
    ]

    # Chain breakdown
    chain_tvls = {k: v for k, v in current_tvl.items()
                  if not k.endswith("-borrowed") and not k.endswith("-staking") and v > 0}
    if chain_tvls:
        lines.append("TVL by Chain:")
        for chain, tvl in sorted(chain_tvls.items(), key=lambda x: x[1], reverse=True)[:10]:
            lines.append(f"  {chain}: ${tvl:,.0f}")
        lines.append("")

    # TVL history (last few data points)
    tvl_history = data.get("tvl", [])
    if tvl_history and len(tvl_history) >= 2:
        latest = tvl_history[-1].get("totalLiquidityUSD", 0)
        week_ago = tvl_history[-min(7, len(tvl_history))].get("totalLiquidityUSD", 0)
        month_ago = tvl_history[-min(30, len(tvl_history))].get("totalLiquidityUSD", 0)

        if week_ago > 0:
            lines.append(f"7d TVL Change: {((latest / week_ago) - 1) * 100:+.2f}%")
        if month_ago > 0:
            lines.append(f"30d TVL Change: {((latest / month_ago) - 1) * 100:+.2f}%")

    return "\n".join(lines)


def get_top_defi_protocols(limit: int = 10) -> str:
    """Get top DeFi protocols ranked by TVL.

    Args:
        limit: Number of protocols to return

    Returns:
        Formatted string with protocol rankings
    """
    try:
        data = http_get_with_retry(
            f"{DEFILLAMA_BASE}/protocols",
            timeout=15,
            retries=2,
        )
    except Exception as e:
        logger.warning("DeFi Llama protocols request failed: %s", e)
        return f"[ERROR] Failed to fetch DeFi protocol rankings: {e}"

    if not data:
        return "[ERROR] No DeFi protocol data available"

    # Sort by TVL descending
    protocols = sorted(data, key=lambda x: x.get("tvl", 0), reverse=True)[:limit]

    lines = [
        f"Top {limit} DeFi Protocols by TVL",
        "",
        "Rank | Protocol | TVL | Category | 1d Change",
        "--- | --- | --- | --- | ---",
    ]

    for i, p in enumerate(protocols, 1):
        tvl = p.get("tvl", 0)
        change = p.get("change_1d", 0)
        lines.append(
            f"{i} | {p.get('name', 'N/A')} | ${tvl:,.0f} | "
            f"{p.get('category', 'N/A')} | {change:+.2f}%" if change else
            f"{i} | {p.get('name', 'N/A')} | ${tvl:,.0f} | "
            f"{p.get('category', 'N/A')} | N/A"
        )

    return "\n".join(lines)


def get_whale_transactions(
    address: Optional[str] = None,
    token: str = "eth",
    min_value_usd: float = 100000,
) -> str:
    """Get recent large transactions (whale activity) from Etherscan.

    Requires ETHERSCAN_API_KEY for higher rate limits (free tier: 5 req/s).

    Args:
        address: Specific address to monitor (or None for general whale activity)
        token: Token to track ("eth" for native ETH)
        min_value_usd: Minimum transaction value in USD to report

    Returns:
        Formatted string with recent large transactions
    """
    api_key = os.environ.get("ETHERSCAN_API_KEY", "")

    if address:
        try:
            data = http_get_with_retry(
                "https://api.etherscan.io/api",
                params={
                    "module": "account",
                    "action": "txlist",
                    "address": address,
                    "startblock": 0,
                    "endblock": 99999999,
                    "page": 1,
                    "offset": 20,
                    "sort": "desc",
                    "apikey": api_key,
                },
                timeout=15,
                retries=2,
            )
        except Exception as e:
            logger.warning("Etherscan request failed: %s", e)
            return f"[ERROR] Failed to fetch transactions for {address}: {e}"

        if data.get("status") != "1":
            return f"[ERROR] Etherscan: {data.get('message', 'Unknown error')}"

        txs = data.get("result", [])
        if not txs:
            return f"No transactions found for address {address}"

        lines = [
            f"Recent Transactions for: {address[:10]}...{address[-6:]}",
            "",
        ]

        for i, tx in enumerate(txs[:10], 1):
            value_eth = int(tx.get("value", 0)) / 1e18
            if value_eth < 0.01:
                continue
            ts = int(tx.get("timeStamp", 0))
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if ts else "N/A"
            direction = "OUT" if tx.get("from", "").lower() == address.lower() else "IN"
            lines.append(f"{i}. [{dt}] {direction} {value_eth:.4f} ETH")
            lines.append(f"   Hash: {tx.get('hash', 'N/A')[:16]}...")
            lines.append("")

        return "\n".join(lines)

    return "[ERROR] Address required for whale transaction lookup. Provide a wallet address to monitor."
