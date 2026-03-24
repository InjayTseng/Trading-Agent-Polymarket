"""Prediction market data access layer.

Provides tools for fetching market data, news, and real-time streaming.
"""

from .polymarket import (
    get_polymarket_market_info,
    get_polymarket_price_history,
    get_polymarket_order_book,
    get_polymarket_resolution_criteria,
    get_polymarket_event_context,
    get_polymarket_related_markets,
    get_polymarket_search,
)
from .news import (
    get_pm_news,
    get_pm_global_news,
)
from .polymarket_ws import PolymarketWebSocket
from .onchain import (
    get_crypto_price,
    get_crypto_price_history,
    get_defi_tvl,
    get_top_defi_protocols,
    get_whale_transactions,
)

__all__ = [
    # Polymarket REST API
    "get_polymarket_market_info",
    "get_polymarket_price_history",
    "get_polymarket_order_book",
    "get_polymarket_resolution_criteria",
    "get_polymarket_event_context",
    "get_polymarket_related_markets",
    "get_polymarket_search",
    # News
    "get_pm_news",
    "get_pm_global_news",
    # Real-time
    "PolymarketWebSocket",
    # On-chain / Crypto
    "get_crypto_price",
    "get_crypto_price_history",
    "get_defi_tvl",
    "get_top_defi_protocols",
    "get_whale_transactions",
]
