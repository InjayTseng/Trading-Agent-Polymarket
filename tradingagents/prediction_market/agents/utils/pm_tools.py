"""Tool definitions for prediction market agents.

Each tool is a @tool-decorated function that calls the Polymarket data layer.
"""

from langchain_core.tools import tool

from tradingagents.prediction_market.dataflows.polymarket import (
    get_polymarket_market_info,
    get_polymarket_price_history,
    get_polymarket_order_book,
    get_polymarket_resolution_criteria,
    get_polymarket_event_context,
    get_polymarket_related_markets,
    get_polymarket_search,
)
from tradingagents.prediction_market.dataflows.news import (
    get_pm_news,
    get_pm_global_news,
)
from tradingagents.prediction_market.dataflows.onchain import (
    get_crypto_price as _get_crypto_price,
    get_crypto_price_history as _get_crypto_price_history,
    get_defi_tvl as _get_defi_tvl,
    get_top_defi_protocols as _get_top_defi_protocols,
)


@tool
def get_market_info(market_id: str) -> str:
    """Get prediction market info including question, current prices, volume, liquidity, and resolution criteria.

    Args:
        market_id: The Polymarket market/condition ID
    """
    return get_polymarket_market_info(market_id)


@tool
def get_market_price_history(market_id: str, start_date: str, end_date: str) -> str:
    """Get historical probability time series for a prediction market.

    Args:
        market_id: The Polymarket market/condition ID
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    return get_polymarket_price_history(market_id, start_date, end_date)


@tool
def get_order_book(market_id: str) -> str:
    """Get current order book depth and spread analysis for a prediction market.

    Args:
        market_id: The Polymarket market/condition ID
    """
    return get_polymarket_order_book(market_id)


@tool
def get_resolution_criteria(market_id: str) -> str:
    """Get detailed resolution criteria, source, and timeline for a prediction market.

    Args:
        market_id: The Polymarket market/condition ID
    """
    return get_polymarket_resolution_criteria(market_id)


@tool
def get_event_context(event_id: str) -> str:
    """Get all markets grouped under a prediction market event.

    Args:
        event_id: The Polymarket event ID
    """
    return get_polymarket_event_context(event_id)


@tool
def get_related_markets(query: str, limit: int = 5) -> str:
    """Search for active prediction market events by topic tag, sorted by volume.

    Args:
        query: Topic tag to filter events (e.g. 'Politics', 'Crypto', 'Sports')
        limit: Maximum number of results (default 5)
    """
    return get_polymarket_related_markets(query, limit)


@tool
def search_markets(query: str, limit: int = 10) -> str:
    """Search Polymarket for markets matching a query string.

    Args:
        query: Search query (e.g. 'US election', 'Bitcoin', 'Fed rate')
        limit: Maximum number of results (default 10)
    """
    return get_polymarket_search(query, limit)


@tool
def get_news(query: str) -> str:
    """Search for news articles relevant to a prediction market question.

    Uses NewsAPI if NEWSAPI_KEY is set, falls back to Yahoo Finance.
    Best for political, sports, crypto, and general event markets.

    Args:
        query: Search query — use the market question or key terms from it
    """
    return get_pm_news(query)


@tool
def get_global_news() -> str:
    """Get top global news headlines relevant to prediction markets.

    Returns the latest headlines that may affect open prediction markets.
    Uses NewsAPI if NEWSAPI_KEY is set, falls back to Yahoo Finance.
    """
    return get_pm_global_news()


@tool
def get_crypto_data(coin_id: str) -> str:
    """Get current price, volume, market cap, and recent changes for a cryptocurrency.

    Useful for crypto-related prediction markets. Uses CoinGecko (free, no API key).

    Args:
        coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum', 'solana', 'cardano')
    """
    return _get_crypto_price(coin_id)


@tool
def get_crypto_history(coin_id: str, days: int = 30) -> str:
    """Get historical price data for a cryptocurrency over a time period.

    Useful for analyzing price trends relevant to crypto prediction markets.

    Args:
        coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
        days: Number of days of history (7, 30, 90, 365)
    """
    return _get_crypto_price_history(coin_id, days=days)


@tool
def get_defi_protocol_tvl(protocol: str) -> str:
    """Get Total Value Locked (TVL) data for a DeFi protocol.

    Useful for DeFi-related prediction markets. Uses DeFi Llama (free, no API key).

    Args:
        protocol: DeFi Llama protocol slug (e.g., 'aave', 'uniswap', 'lido', 'maker')
    """
    return _get_defi_tvl(protocol)


@tool
def get_top_defi(limit: int = 10) -> str:
    """Get top DeFi protocols ranked by Total Value Locked.

    Provides an overview of the DeFi landscape for crypto prediction markets.

    Args:
        limit: Number of protocols to return (default 10)
    """
    return _get_top_defi_protocols(limit=limit)
