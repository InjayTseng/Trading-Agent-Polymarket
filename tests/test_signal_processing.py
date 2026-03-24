"""Tests for PMSignalProcessor regex-based signal extraction."""

import json
import pytest

from tradingagents.prediction_market.graph.signal_processing import (
    PMSignalProcessor,
    PASS_SIGNAL,
)


@pytest.fixture
def processor():
    return PMSignalProcessor()


def parse(processor, text: str) -> dict:
    return json.loads(processor.process_signal(text))


# ── Empty / None input ──────────────────────────────────────────────

class TestEmptyInput:
    def test_none_returns_pass(self, processor):
        result = parse(processor, None)
        assert result["signal"] == "PASS"

    def test_empty_string_returns_pass(self, processor):
        result = parse(processor, "")
        assert result["signal"] == "PASS"

    def test_pass_signal_constant_structure(self):
        result = json.loads(PASS_SIGNAL)
        assert result["signal"] == "PASS"
        for key in ("estimated_probability", "market_price", "edge", "position_size", "confidence"):
            assert result[key] is None


# ── Signal extraction (BUY_YES / BUY_NO / PASS) ────────────────────

class TestSignalExtraction:
    def test_final_trade_decision_buy_yes(self, processor):
        text = "After analysis, FINAL TRADE DECISION: BUY_YES with high confidence."
        assert parse(processor, text)["signal"] == "BUY_YES"

    def test_final_trade_decision_buy_no(self, processor):
        text = "FINAL TRADE DECISION: BUY_NO"
        assert parse(processor, text)["signal"] == "BUY_NO"

    def test_final_trade_decision_pass(self, processor):
        text = "FINAL TRADE DECISION: PASS"
        assert parse(processor, text)["signal"] == "PASS"

    def test_final_trade_proposal(self, processor):
        text = "FINAL TRADE PROPOSAL: **BUY_YES**"
        assert parse(processor, text)["signal"] == "BUY_YES"

    def test_recommendation_pattern(self, processor):
        text = "RECOMMENDATION: **BUY_NO**"
        assert parse(processor, text)["signal"] == "BUY_NO"

    def test_recommendation_no_bold(self, processor):
        text = "Recommendation: BUY_YES"
        assert parse(processor, text)["signal"] == "BUY_YES"

    def test_fallback_keyword_buy_yes(self, processor):
        text = "I recommend we go with BUY_YES on this market."
        assert parse(processor, text)["signal"] == "BUY_YES"

    def test_fallback_keyword_buy_no(self, processor):
        text = "The best course of action is BUY_NO."
        assert parse(processor, text)["signal"] == "BUY_NO"

    def test_no_signal_defaults_pass(self, processor):
        text = "The market looks interesting but I have no strong opinion."
        assert parse(processor, text)["signal"] == "PASS"

    def test_final_decision_takes_precedence_over_keyword(self, processor):
        """If both FINAL TRADE DECISION and keyword appear, the formal pattern wins."""
        text = "Earlier I considered BUY_YES but FINAL TRADE DECISION: BUY_NO"
        assert parse(processor, text)["signal"] == "BUY_NO"


# ── Estimated probability extraction ────────────────────────────────

class TestProbabilityExtraction:
    def test_percentage_format(self, processor):
        text = "Estimated probability: 65%. FINAL TRADE DECISION: BUY_YES"
        result = parse(processor, text)
        assert result["estimated_probability"] == 0.65

    def test_decimal_format(self, processor):
        text = "Estimated true probability is 0.72. FINAL TRADE DECISION: BUY_YES"
        result = parse(processor, text)
        assert result["estimated_probability"] == 0.72

    def test_approximate_phrasing(self, processor):
        text = "True probability is approximately 58%. BUY_YES"
        result = parse(processor, text)
        assert result["estimated_probability"] == 0.58

    def test_my_estimate_phrasing(self, processor):
        text = "My estimate is 70%. BUY_YES"
        result = parse(processor, text)
        assert result["estimated_probability"] == 0.70

    def test_no_probability_returns_none(self, processor):
        text = "FINAL TRADE DECISION: BUY_YES"
        result = parse(processor, text)
        assert result["estimated_probability"] is None


# ── Market price extraction ─────────────────────────────────────────

class TestMarketPriceExtraction:
    def test_decimal_format(self, processor):
        text = "Market price is 0.55. BUY_YES"
        result = parse(processor, text)
        assert result["market_price"] == 0.55

    def test_percentage_format(self, processor):
        text = "Current market price: 62%. BUY_YES"
        result = parse(processor, text)
        assert result["market_price"] == 0.62

    def test_dollar_sign(self, processor):
        text = "Current price is $0.48. BUY_YES"
        result = parse(processor, text)
        assert result["market_price"] == 0.48

    def test_approximately(self, processor):
        text = "Market price is approximately 0.60. BUY_YES"
        result = parse(processor, text)
        assert result["market_price"] == 0.60

    def test_no_price_returns_none(self, processor):
        text = "FINAL TRADE DECISION: BUY_YES"
        result = parse(processor, text)
        assert result["market_price"] is None


# ── Edge extraction ─────────────────────────────────────────────────

class TestEdgeExtraction:
    def test_percentage(self, processor):
        text = "Edge is 12%. BUY_YES"
        result = parse(processor, text)
        assert result["edge"] == 0.12

    def test_decimal(self, processor):
        text = "Perceived edge: 0.08. BUY_YES"
        result = parse(processor, text)
        assert result["edge"] == 0.08

    def test_computed_from_probability_and_price(self, processor):
        text = "Estimated probability: 70%. Market price is 0.55. BUY_YES"
        result = parse(processor, text)
        assert result["edge"] == pytest.approx(0.15, abs=0.01)

    def test_explicit_edge_takes_priority(self, processor):
        text = "Estimated probability: 70%. Market price is 0.55. Edge is 10%. BUY_YES"
        result = parse(processor, text)
        assert result["edge"] == 0.10

    def test_no_edge_no_components_returns_none(self, processor):
        text = "FINAL TRADE DECISION: BUY_YES"
        result = parse(processor, text)
        assert result["edge"] is None


# ── Position size extraction ────────────────────────────────────────

class TestPositionSizeExtraction:
    def test_decimal(self, processor):
        text = "Position size: 0.03. BUY_YES"
        result = parse(processor, text)
        assert result["position_size"] == 0.03

    def test_percentage(self, processor):
        text = "Position size is 4%. BUY_YES"
        result = parse(processor, text)
        assert result["position_size"] == 0.04

    def test_capped_at_ten_percent(self, processor):
        text = "Position size: 0.25. BUY_YES"
        result = parse(processor, text)
        assert result["position_size"] == 0.10

    def test_sizing_keyword(self, processor):
        text = "Sizing = 0.05. BUY_YES"
        result = parse(processor, text)
        assert result["position_size"] == 0.05

    def test_no_size_returns_none(self, processor):
        text = "FINAL TRADE DECISION: BUY_YES"
        result = parse(processor, text)
        assert result["position_size"] is None


# ── Confidence extraction ───────────────────────────────────────────

class TestConfidenceExtraction:
    def test_high(self, processor):
        text = "Confidence level: high. BUY_YES"
        result = parse(processor, text)
        assert result["confidence"] == "high"

    def test_medium(self, processor):
        text = "Confidence: medium. BUY_YES"
        result = parse(processor, text)
        assert result["confidence"] == "medium"

    def test_low(self, processor):
        text = "Confidence is low. BUY_NO"
        result = parse(processor, text)
        assert result["confidence"] == "low"

    def test_case_insensitive(self, processor):
        text = "Confidence level: HIGH. BUY_YES"
        result = parse(processor, text)
        assert result["confidence"] == "high"

    def test_no_confidence_returns_none(self, processor):
        text = "FINAL TRADE DECISION: BUY_YES"
        result = parse(processor, text)
        assert result["confidence"] is None


# ── Full realistic signals ──────────────────────────────────────────

class TestFullSignals:
    def test_complete_buy_yes_signal(self, processor):
        text = """
        Based on my analysis of the market data and research debate:

        Estimated true probability: 72%
        Current market price: 0.58
        Edge is 14%
        Position size: 0.035
        Confidence level: high

        FINAL TRADE DECISION: **BUY_YES**
        """
        result = parse(processor, text)
        assert result["signal"] == "BUY_YES"
        assert result["estimated_probability"] == 0.72
        assert result["market_price"] == 0.58
        assert result["edge"] == 0.14
        assert result["position_size"] == 0.035
        assert result["confidence"] == "high"

    def test_complete_pass_signal(self, processor):
        text = """
        The edge is too small to justify a position.

        Estimated probability: 53%
        Market price is 0.51
        Edge: 2%

        FINAL TRADE DECISION: PASS
        """
        result = parse(processor, text)
        assert result["signal"] == "PASS"
        assert result["estimated_probability"] == 0.53
        assert result["edge"] == 0.02

    def test_buy_no_with_recommendation(self, processor):
        text = """
        The market is overpriced.
        My estimate is 30%.
        Market price: 0.52
        Perceived edge: 0.22
        Position size: 0.04
        Confidence: medium
        Recommendation: BUY_NO
        """
        result = parse(processor, text)
        assert result["signal"] == "BUY_NO"
        assert result["estimated_probability"] == 0.30
        assert result["market_price"] == 0.52
        assert result["edge"] == 0.22
        assert result["position_size"] == 0.04
        assert result["confidence"] == "medium"

    def test_output_is_valid_json(self, processor):
        text = "FINAL TRADE DECISION: BUY_YES. Confidence: high."
        raw = processor.process_signal(text)
        result = json.loads(raw)
        assert isinstance(result, dict)
        expected_keys = {"signal", "estimated_probability", "market_price", "edge", "position_size", "confidence"}
        assert set(result.keys()) == expected_keys
