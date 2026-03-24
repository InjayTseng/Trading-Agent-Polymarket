"""News fetching for prediction market analysis.

Supports multiple backends:
1. NewsAPI (newsapi.org) — best for political, sports, and general event markets
2. yfinance — fallback for financial/economic markets
3. Built-in web search — zero-config fallback

Set NEWSAPI_KEY in .env for the best experience.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

from .http_utils import http_get_with_retry

logger = logging.getLogger(__name__)

# Try to import optional dependencies
_HAS_YFINANCE = False

try:
    import yfinance as _yf
    _HAS_YFINANCE = True
except ImportError:
    pass

NEWSAPI_BASE = "https://newsapi.org/v2"


def _newsapi_search(
    query: str,
    days_back: int = 7,
    limit: int = 10,
    language: str = "en",
) -> Optional[str]:
    """Search for news articles using NewsAPI.org.

    Requires NEWSAPI_KEY environment variable.
    Free tier: 100 requests/day, last 30 days of articles.
    """
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        return None

    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    try:
        data = http_get_with_retry(
            f"{NEWSAPI_BASE}/everything",
            params={
                "q": query,
                "from": from_date,
                "sortBy": "relevancy",
                "pageSize": limit,
                "language": language,
                "apiKey": api_key,
            },
            timeout=15,
            retries=2,
        )
    except Exception as e:
        logger.warning("NewsAPI request failed: %s", e)
        return None

    articles = data.get("articles", [])
    if not articles:
        return None

    lines = [
        f"News results for: '{query}' (last {days_back} days, via NewsAPI)",
        f"Total results: {data.get('totalResults', 0)}",
        "",
    ]

    for i, article in enumerate(articles[:limit], 1):
        title = article.get("title", "N/A")
        source = article.get("source", {}).get("name", "Unknown")
        published = article.get("publishedAt", "")[:10]
        description = article.get("description", "")
        lines.append(f"{i}. [{published}] {title}")
        lines.append(f"   Source: {source}")
        if description:
            lines.append(f"   {description[:200]}")
        lines.append("")

    return "\n".join(lines)


def _yfinance_news(query: str, limit: int = 10) -> Optional[str]:
    """Search for news using yfinance (best for financial topics)."""
    if not _HAS_YFINANCE:
        return None

    try:
        # yfinance news search via ticker or search
        ticker = _yf.Ticker(query)
        news = ticker.news
        if not news:
            # Try as a search term
            search = _yf.Search(query)
            news = getattr(search, "news", []) or []
    except Exception as e:
        logger.debug("yfinance news lookup failed for '%s': %s", query, e)
        return None

    if not news:
        return None

    lines = [
        f"Financial news for: '{query}' (via Yahoo Finance)",
        "",
    ]

    for i, item in enumerate(news[:limit], 1):
        title = item.get("title", "N/A")
        publisher = item.get("publisher", "Unknown")
        lines.append(f"{i}. {title}")
        lines.append(f"   Source: {publisher}")
        lines.append("")

    return "\n".join(lines)


def get_pm_news(
    query: str,
    days_back: int = 7,
    limit: int = 10,
    language: str = "en",
) -> str:
    """Get news relevant to a prediction market question.

    Tries NewsAPI first (best for PM topics), falls back to yfinance.

    Args:
        query: Search query (typically the market question or key terms)
        days_back: How many days back to search
        limit: Maximum number of articles
        language: Language code for NewsAPI (e.g., "en", "de", "fr")

    Returns:
        Formatted string of news results
    """
    # Try NewsAPI first (best for political/general markets)
    result = _newsapi_search(query, days_back=days_back, limit=limit, language=language)
    if result:
        return result

    # Fall back to yfinance (better for financial markets)
    result = _yfinance_news(query, limit=limit)
    if result:
        return result

    return f"No news found for: '{query}'. Consider setting NEWSAPI_KEY in .env for better news coverage."


def get_pm_global_news(limit: int = 10, country: str = "us") -> str:
    """Get top global news headlines relevant to prediction markets.

    Tries NewsAPI top headlines, falls back to yfinance.

    Args:
        limit: Maximum number of articles
        country: Country code for top headlines (e.g., "us", "gb", "de")

    Returns:
        Formatted string of top headlines
    """
    api_key = os.environ.get("NEWSAPI_KEY")
    if api_key:
        try:
            data = http_get_with_retry(
                f"{NEWSAPI_BASE}/top-headlines",
                params={
                    "country": country,
                    "pageSize": limit,
                    "apiKey": api_key,
                },
                timeout=15,
                retries=2,
            )
            articles = data.get("articles", [])

            if articles:
                lines = [
                    f"Top {limit} Headlines ({country.upper()}, via NewsAPI)",
                    "",
                ]
                for i, article in enumerate(articles[:limit], 1):
                    title = article.get("title", "N/A")
                    source = article.get("source", {}).get("name", "Unknown")
                    published = article.get("publishedAt", "")[:10]
                    lines.append(f"{i}. [{published}] {title}")
                    lines.append(f"   Source: {source}")
                    lines.append("")
                return "\n".join(lines)
        except Exception as e:
            logger.warning("NewsAPI top headlines failed: %s", e)

    # Fallback
    result = _yfinance_news("world news", limit=limit)
    if result:
        return result

    return "No global news available. Set NEWSAPI_KEY in .env for news coverage."
