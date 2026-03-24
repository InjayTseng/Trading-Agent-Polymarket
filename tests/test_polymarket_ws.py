"""Tests for PolymarketWebSocket — book handling, subscriptions, message processing."""

import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.prediction_market.dataflows.polymarket_ws import (
    PolymarketWebSocket,
    MAX_ASSETS_PER_CONNECTION,
)


@pytest.fixture
def ws():
    return PolymarketWebSocket(asset_ids=["token_yes", "token_no"])


# ── Book snapshot handling ──────────────────────────────────────────

class TestBookSnapshot:
    def test_book_event_populates_books(self, ws):
        msg = json.dumps({
            "event_type": "book",
            "asset_id": "token_yes",
            "market": "0xabc",
            "bids": [
                {"price": "0.48", "size": "30"},
                {"price": "0.47", "size": "50"},
            ],
            "asks": [
                {"price": "0.52", "size": "25"},
                {"price": "0.53", "size": "60"},
            ],
        })
        ws._on_message(None, msg)
        assert "token_yes" in ws.books
        assert ws.books["token_yes"]["bids"][Decimal("0.48")] == Decimal("30")
        assert ws.books["token_yes"]["asks"][Decimal("0.52")] == Decimal("25")

    def test_book_replaces_previous_state(self, ws):
        # First snapshot
        ws._on_message(None, json.dumps({
            "event_type": "book",
            "asset_id": "token_yes",
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        }))
        # Second snapshot replaces
        ws._on_message(None, json.dumps({
            "event_type": "book",
            "asset_id": "token_yes",
            "bids": [{"price": "0.48", "size": "30"}],
            "asks": [{"price": "0.52", "size": "25"}],
        }))
        assert Decimal("0.50") not in ws.books["token_yes"]["bids"]
        assert Decimal("0.48") in ws.books["token_yes"]["bids"]

    def test_on_book_callback(self, ws):
        callback = MagicMock()
        ws.on_book = callback
        ws._on_message(None, json.dumps({
            "event_type": "book",
            "asset_id": "token_yes",
            "bids": [], "asks": [],
        }))
        callback.assert_called_once()


# ── Incremental price_change ────────────────────────────────────────

class TestPriceChange:
    def _setup_book(self, ws):
        ws._on_message(None, json.dumps({
            "event_type": "book",
            "asset_id": "token_yes",
            "bids": [{"price": "0.48", "size": "30"}],
            "asks": [{"price": "0.52", "size": "25"}],
        }))

    def test_update_existing_level(self, ws):
        self._setup_book(ws)
        ws._on_message(None, json.dumps({
            "event_type": "price_change",
            "price_changes": [{
                "asset_id": "token_yes",
                "price": "0.48", "size": "50", "side": "BUY",
            }],
        }))
        assert ws.books["token_yes"]["bids"][Decimal("0.48")] == Decimal("50")

    def test_add_new_level(self, ws):
        self._setup_book(ws)
        ws._on_message(None, json.dumps({
            "event_type": "price_change",
            "price_changes": [{
                "asset_id": "token_yes",
                "price": "0.49", "size": "20", "side": "BUY",
            }],
        }))
        assert Decimal("0.49") in ws.books["token_yes"]["bids"]

    def test_size_zero_removes_level(self, ws):
        self._setup_book(ws)
        ws._on_message(None, json.dumps({
            "event_type": "price_change",
            "price_changes": [{
                "asset_id": "token_yes",
                "price": "0.48", "size": "0", "side": "BUY",
            }],
        }))
        assert Decimal("0.48") not in ws.books["token_yes"]["bids"]

    def test_unknown_asset_ignored(self, ws):
        ws._on_message(None, json.dumps({
            "event_type": "price_change",
            "price_changes": [{
                "asset_id": "unknown_token",
                "price": "0.50", "size": "10", "side": "BUY",
            }],
        }))
        assert "unknown_token" not in ws.books


# ── get_book ────────────────────────────────────────────────────────

class TestGetBook:
    def test_returns_sorted_book(self, ws):
        ws._on_message(None, json.dumps({
            "event_type": "book",
            "asset_id": "token_yes",
            "bids": [
                {"price": "0.47", "size": "50"},
                {"price": "0.48", "size": "30"},
                {"price": "0.46", "size": "10"},
            ],
            "asks": [
                {"price": "0.53", "size": "60"},
                {"price": "0.52", "size": "25"},
            ],
        }))
        book = ws.get_book("token_yes")
        # Bids descending
        assert book["bids"][0][0] == "0.48"
        assert book["bids"][-1][0] == "0.46"
        # Asks ascending
        assert book["asks"][0][0] == "0.52"
        assert book["asks"][-1][0] == "0.53"

    def test_filters_zero_size(self, ws):
        ws.books["token_yes"] = {
            "bids": {Decimal("0.48"): Decimal("30"), Decimal("0.47"): Decimal("0")},
            "asks": {Decimal("0.52"): Decimal("25")},
        }
        book = ws.get_book("token_yes")
        assert len(book["bids"]) == 1

    def test_unknown_asset_returns_none(self, ws):
        assert ws.get_book("nonexistent") is None


# ── get_mid_price ───────────────────────────────────────────────────

class TestGetMidPrice:
    def test_mid_price_calculation(self, ws):
        ws.books["token_yes"] = {
            "bids": {Decimal("0.48"): Decimal("30")},
            "asks": {Decimal("0.52"): Decimal("25")},
        }
        mid = ws.get_mid_price("token_yes")
        assert mid == pytest.approx(0.50)

    def test_empty_book_returns_none(self, ws):
        ws.books["token_yes"] = {"bids": {}, "asks": {Decimal("0.52"): Decimal("25")}}
        assert ws.get_mid_price("token_yes") is None

    def test_unknown_asset_returns_none(self, ws):
        assert ws.get_mid_price("nonexistent") is None


# ── Subscribe / Unsubscribe ─────────────────────────────────────────

class TestSubscription:
    def test_initial_asset_ids(self, ws):
        assert ws.asset_ids == ["token_yes", "token_no"]

    def test_subscribe_adds_ids(self, ws):
        ws.subscribe(["token_new"])
        assert "token_new" in ws.asset_ids

    def test_subscribe_deduplicates(self, ws):
        ws.subscribe(["token_yes"])
        assert ws.asset_ids.count("token_yes") == 1

    def test_subscribe_respects_max_limit(self):
        ws = PolymarketWebSocket(asset_ids=[f"t{i}" for i in range(MAX_ASSETS_PER_CONNECTION - 1)])
        ws.subscribe([f"new{i}" for i in range(5)])
        assert len(ws.asset_ids) <= MAX_ASSETS_PER_CONNECTION

    def test_unsubscribe_removes_ids(self, ws):
        ws.unsubscribe(["token_yes"])
        assert "token_yes" not in ws.asset_ids

    def test_unsubscribe_cleans_book(self, ws):
        ws.books["token_yes"] = {"bids": {}, "asks": {}}
        ws.unsubscribe(["token_yes"])
        assert "token_yes" not in ws.books

    def test_subscribe_sends_message_when_connected(self, ws):
        ws._connected.set()
        ws._ws = MagicMock()
        ws.subscribe(["token_new"])
        ws._ws.send.assert_called_once()
        sent = json.loads(ws._ws.send.call_args[0][0])
        assert sent["type"] == "market"
        assert sent["operation"] == "subscribe"
        assert "token_new" in sent["assets_ids"]


# ── Message routing ─────────────────────────────────────────────────

class TestMessageRouting:
    def test_pong_ignored(self, ws):
        ws._on_message(None, "PONG")  # should not raise

    def test_non_json_ignored(self, ws):
        ws._on_message(None, "not json")  # should not raise

    def test_trade_callback(self, ws):
        callback = MagicMock()
        ws.on_trade = callback
        ws._on_message(None, json.dumps({
            "event_type": "last_trade_price",
            "asset_id": "token_yes",
            "price": "0.52",
            "side": "BUY",
            "size": "100",
        }))
        callback.assert_called_once()

    def test_market_resolved_cleans_book(self, ws):
        ws.books["token_yes"] = {"bids": {}, "asks": {}}
        ws._on_message(None, json.dumps({
            "event_type": "market_resolved",
            "asset_id": "token_yes",
            "winning_outcome": "Yes",
        }))
        assert "token_yes" not in ws.books

    def test_unknown_event_type_ignored(self, ws):
        ws._on_message(None, json.dumps({"event_type": "unknown_event"}))
