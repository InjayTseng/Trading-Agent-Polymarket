"""Shared HTTP utilities for dataflow modules."""

import logging
import time

import requests
from requests.exceptions import ConnectionError, Timeout

logger = logging.getLogger(__name__)


def http_get_with_retry(url, params=None, timeout=30, retries=3, backoff_base=1.5):
    """HTTP GET with exponential backoff retry and 429 rate-limit handling.

    Args:
        url: Request URL
        params: Query parameters
        timeout: Request timeout in seconds
        retries: Maximum retry attempts
        backoff_base: Base for exponential backoff

    Returns:
        Parsed JSON response

    Raises:
        RuntimeError: After all retries exhausted
    """
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", backoff_base ** attempt))
                logger.warning("Rate limited by %s, retrying in %ds", url, retry_after)
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            return resp.json()
        except (ConnectionError, Timeout) as e:
            if attempt == retries - 1:
                raise
            wait = backoff_base ** attempt
            logger.warning(
                "Request to %s failed (attempt %d/%d): %s. Retrying in %.1fs",
                url, attempt + 1, retries, e, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} retries: {url}")
