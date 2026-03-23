# TradingAgents/prediction_market/graph/signal_processing.py

import json
import re
import logging

logger = logging.getLogger(__name__)

PASS_SIGNAL = json.dumps({
    "signal": "PASS",
    "estimated_probability": None,
    "market_price": None,
    "edge": None,
    "position_size": None,
    "confidence": None,
})


class PMSignalProcessor:
    """Processes prediction market trading signals using deterministic regex parsing.

    No LLM call required — extracts structured data directly from the Risk Judge
    or Trader's natural language output using pattern matching.
    """

    def __init__(self, quick_thinking_llm=None):
        """Initialize. LLM param kept for backward compatibility but is not used."""
        pass

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full prediction market trading signal to extract the core decision
        and structured data using regex parsing.

        Args:
            full_signal: Complete trading signal text from the risk manager or trader

        Returns:
            JSON string with signal, estimated_probability, market_price, edge,
            position_size, and confidence
        """
        if not full_signal:
            logger.warning("Empty signal received, defaulting to PASS")
            return PASS_SIGNAL

        upper_signal = full_signal.upper()

        # Extract signal (BUY_YES, BUY_NO, or PASS)
        signal = "PASS"
        # Check for explicit final decision markers first
        final_patterns = [
            r"FINAL\s+TRADE\s+DECISION\s*:\s*\*?\*?\s*(BUY_YES|BUY_NO|PASS)\s*\*?\*?",
            r"FINAL\s+TRADE\s+PROPOSAL\s*:\s*\*?\*?\s*(BUY_YES|BUY_NO|PASS)\s*\*?\*?",
            r"RECOMMENDATION\s*:\s*\*?\*?\s*(BUY_YES|BUY_NO|PASS)\s*\*?\*?",
        ]
        for pattern in final_patterns:
            match = re.search(pattern, upper_signal)
            if match:
                signal = match.group(1)
                break
        else:
            # Fallback: scan for keywords
            if "BUY_YES" in upper_signal:
                signal = "BUY_YES"
            elif "BUY_NO" in upper_signal:
                signal = "BUY_NO"

        # Extract estimated probability (look for patterns like "65%", "0.65", "estimated probability")
        estimated_probability = None
        prob_patterns = [
            r"(?:estimated?\s+(?:true\s+)?probability|true\s+probability|my\s+estimate)\s*(?:is|:)\s*(?:approximately?\s+)?(\d{1,3}(?:\.\d+)?)\s*%",
            r"(?:estimated?\s+(?:true\s+)?probability|true\s+probability)\s*(?:is|:)\s*(?:approximately?\s+)?(0\.\d+)",
        ]
        for pattern in prob_patterns:
            match = re.search(pattern, full_signal, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                estimated_probability = val / 100 if val > 1 else val
                break

        # Extract market price
        market_price = None
        price_patterns = [
            r"(?:market\s+price|current\s+(?:market\s+)?price|market\s+(?:is\s+)?(?:priced?\s+)?(?:at)?)\s*(?:is|:)?\s*(?:approximately?\s+)?(?:\$)?(0\.\d+)",
            r"(?:market\s+price|current\s+price)\s*(?:is|:)?\s*(?:approximately?\s+)?(\d{1,3}(?:\.\d+)?)\s*%",
        ]
        for pattern in price_patterns:
            match = re.search(pattern, full_signal, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                market_price = val / 100 if val > 1 else val
                break

        # Extract edge
        edge = None
        edge_patterns = [
            r"(?:edge|perceived\s+edge)\s*(?:is|:|\=)\s*(?:approximately?\s+)?(\d{1,3}(?:\.\d+)?)\s*%",
            r"(?:edge|perceived\s+edge)\s*(?:is|:|\=)\s*(?:approximately?\s+)?(0\.\d+)",
        ]
        for pattern in edge_patterns:
            match = re.search(pattern, full_signal, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                edge = val / 100 if val > 1 else val
                break
        # Compute edge from probability and price if not found directly
        if edge is None and estimated_probability is not None and market_price is not None:
            edge = round(abs(estimated_probability - market_price), 4)

        # Extract position size
        position_size = None
        size_patterns = [
            r"(?:position\s+size|sizing)\s*(?:is|:|\=)\s*(?:approximately?\s+)?(0\.\d+)",
            r"(?:position\s+size|sizing)\s*(?:is|:|\=)\s*(?:approximately?\s+)?(\d{1,2}(?:\.\d+)?)\s*%",
        ]
        for pattern in size_patterns:
            match = re.search(pattern, full_signal, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                position_size = val / 100 if val > 1 else val
                break
        # Cap position size
        if position_size is not None:
            position_size = min(position_size, 0.10)

        # Extract confidence
        confidence = None
        conf_match = re.search(r"(?:confidence)\s*(?:level|:|\s+is)\s*(?::?\s*)?(low|medium|high)", full_signal, re.IGNORECASE)
        if conf_match:
            confidence = conf_match.group(1).lower()

        result = {
            "signal": signal,
            "estimated_probability": estimated_probability,
            "market_price": market_price,
            "edge": edge,
            "position_size": position_size,
            "confidence": confidence,
        }

        return json.dumps(result)
