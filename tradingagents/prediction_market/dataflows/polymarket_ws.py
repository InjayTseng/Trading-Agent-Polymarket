"""Polymarket WebSocket client for real-time market data streaming.

Connects to the public Market channel (no authentication required) for:
- Order book snapshots and incremental updates
- Trade executions (last_trade_price)
- Best bid/ask changes
- Market resolution notifications

Usage:
    client = PolymarketWebSocket(asset_ids=["token_id_1", "token_id_2"])
    client.on_trade = lambda data: print(data)
    client.connect()  # blocking
    # or
    client.connect_async()  # returns Thread
"""

import json
import logging
import threading
import time
from typing import Callable, Dict, List, Optional
from decimal import Decimal

try:
    import websocket
except ImportError:
    websocket = None

logger = logging.getLogger(__name__)

WS_MARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
PING_INTERVAL = 10  # seconds
MAX_ASSETS_PER_CONNECTION = 500


class PolymarketWebSocket:
    """Real-time market data client using the Polymarket CLOB WebSocket API.

    Subscribes to the Market channel for order book, trade, and price events.
    All values from the API are strings — this client parses them into native types.
    """

    def __init__(
        self,
        asset_ids: Optional[List[str]] = None,
        url: str = WS_MARKET_URL,
        custom_features: bool = True,
        auto_reconnect: bool = True,
        max_reconnect_delay: float = 60.0,
    ):
        if websocket is None:
            raise ImportError(
                "websocket-client is required for WebSocket support. "
                "Install it with: pip install websocket-client"
            )

        self.url = url
        self.asset_ids: List[str] = list(asset_ids or [])
        self.custom_features = custom_features
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_delay = max_reconnect_delay

        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._ping_thread: Optional[threading.Thread] = None
        self._connected = threading.Event()
        self._stop = threading.Event()
        self._reconnect_count = 0

        # Order book state: asset_id -> {"bids": {price: size}, "asks": {price: size}}
        self.books: Dict[str, dict] = {}

        # Callbacks — assign your handlers
        self.on_book: Optional[Callable[[dict], None]] = None
        self.on_price_change: Optional[Callable[[dict], None]] = None
        self.on_trade: Optional[Callable[[dict], None]] = None
        self.on_best_bid_ask: Optional[Callable[[dict], None]] = None
        self.on_tick_size_change: Optional[Callable[[dict], None]] = None
        self.on_market_resolved: Optional[Callable[[dict], None]] = None
        self.on_new_market: Optional[Callable[[dict], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def connect(self):
        """Connect and block until stopped."""
        self._stop.clear()
        self._run()

    def connect_async(self) -> threading.Thread:
        """Connect in a background thread. Returns the thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="polymarket-ws")
        self._thread.start()
        self._connected.wait(timeout=10)
        return self._thread

    def disconnect(self):
        """Gracefully disconnect."""
        self._stop.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def subscribe(self, asset_ids: List[str]):
        """Dynamically subscribe to additional assets without reconnecting."""
        if not asset_ids:
            return
        new_ids = [aid for aid in asset_ids if aid not in self.asset_ids]
        if not new_ids:
            return
        if len(self.asset_ids) + len(new_ids) > MAX_ASSETS_PER_CONNECTION:
            logger.warning(
                "Adding %d assets would exceed %d limit. Truncating.",
                len(new_ids), MAX_ASSETS_PER_CONNECTION,
            )
            new_ids = new_ids[:MAX_ASSETS_PER_CONNECTION - len(self.asset_ids)]

        self.asset_ids.extend(new_ids)
        if self.is_connected and self._ws:
            msg = json.dumps({
                "assets_ids": new_ids,
                "type": "market",
                "operation": "subscribe",
                "custom_feature_enabled": self.custom_features,
            })
            self._ws.send(msg)
            logger.info("Subscribed to %d additional assets", len(new_ids))

    def unsubscribe(self, asset_ids: List[str]):
        """Dynamically unsubscribe from assets."""
        if not asset_ids:
            return
        self.asset_ids = [aid for aid in self.asset_ids if aid not in asset_ids]
        for aid in asset_ids:
            self.books.pop(aid, None)
        if self.is_connected and self._ws:
            msg = json.dumps({"assets_ids": asset_ids, "operation": "unsubscribe"})
            self._ws.send(msg)
            logger.info("Unsubscribed from %d assets", len(asset_ids))

    def get_book(self, asset_id: str) -> Optional[dict]:
        """Get the current order book snapshot for an asset.

        Returns dict with 'bids' and 'asks' as lists of (price, size) tuples
        sorted by price (bids descending, asks ascending).
        """
        book = self.books.get(asset_id)
        if not book:
            return None
        bids = sorted(book["bids"].items(), key=lambda x: x[0], reverse=True)
        asks = sorted(book["asks"].items(), key=lambda x: x[0])
        return {
            "bids": [(str(p), str(s)) for p, s in bids if s > 0],
            "asks": [(str(p), str(s)) for p, s in asks if s > 0],
        }

    def get_mid_price(self, asset_id: str) -> Optional[float]:
        """Get the midpoint price for an asset (O(1) from cached best levels)."""
        book = self.books.get(asset_id)
        if not book or not book["bids"] or not book["asks"]:
            return None
        best_bid = max(book["bids"])
        best_ask = min(book["asks"])
        return float((best_bid + best_ask) / 2)

    # ── Internal ────────────────────────────────────────────────

    def _run(self):
        while not self._stop.is_set():
            try:
                self._ws = websocket.WebSocketApp(
                    self.url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=0)  # we handle pings manually
            except Exception as e:
                logger.error("WebSocket run error: %s", e)

            self._connected.clear()

            if self._stop.is_set() or not self.auto_reconnect:
                break

            delay = min(2 ** self._reconnect_count, self.max_reconnect_delay)
            self._reconnect_count += 1
            logger.info("Reconnecting in %.1fs (attempt %d)", delay, self._reconnect_count)
            self._stop.wait(timeout=delay)

    def _on_open(self, ws):
        logger.info("Connected to %s", self.url)
        self._reconnect_count = 0

        # Must send subscription immediately or server drops the connection
        if self.asset_ids:
            msg = json.dumps({
                "assets_ids": self.asset_ids[:MAX_ASSETS_PER_CONNECTION],
                "type": "market",
                "custom_feature_enabled": self.custom_features,
            })
            ws.send(msg)
            logger.info("Subscribed to %d assets", min(len(self.asset_ids), MAX_ASSETS_PER_CONNECTION))

        self._connected.set()

        # Start heartbeat thread
        self._ping_thread = threading.Thread(target=self._heartbeat, daemon=True, name="polymarket-ws-ping")
        self._ping_thread.start()

    def _heartbeat(self):
        while self.is_connected and not self._stop.is_set():
            try:
                if self._ws:
                    self._ws.send("PING")
            except Exception:
                break
            self._stop.wait(timeout=PING_INTERVAL)

    def _on_message(self, ws, message: str):
        if message == "PONG":
            return

        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("Non-JSON message: %s", message[:100])
            return

        event_type = data.get("event_type")
        if not event_type:
            return

        handler_map = {
            "book": self._handle_book,
            "price_change": self._handle_price_change,
            "last_trade_price": self._handle_trade,
            "best_bid_ask": self._handle_best_bid_ask,
            "tick_size_change": self._handle_tick_size_change,
            "market_resolved": self._handle_market_resolved,
            "new_market": self._handle_new_market,
        }

        handler = handler_map.get(event_type)
        if handler:
            try:
                handler(data)
            except Exception as e:
                logger.error("Error handling %s event: %s", event_type, e)

    def _on_error(self, ws, error):
        logger.error("WebSocket error: %s", error)
        if self.on_error:
            self.on_error(error)

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket closed (code=%s, msg=%s)", close_status_code, close_msg)
        self._connected.clear()

    # ── Event handlers ──────────────────────────────────────────

    def _handle_book(self, data: dict):
        """Process full book snapshot — replaces local state."""
        asset_id = data.get("asset_id")
        if not asset_id:
            return

        bids = {}
        for entry in data.get("bids", []):
            price = Decimal(entry["price"])
            size = Decimal(entry["size"])
            bids[price] = size

        asks = {}
        for entry in data.get("asks", []):
            price = Decimal(entry["price"])
            size = Decimal(entry["size"])
            asks[price] = size

        self.books[asset_id] = {"bids": bids, "asks": asks}

        if self.on_book:
            self.on_book(data)

    def _handle_price_change(self, data: dict):
        """Apply incremental book update. size=0 removes the level."""
        for change in data.get("price_changes", []):
            asset_id = change.get("asset_id")
            if not asset_id or asset_id not in self.books:
                continue

            price = Decimal(change["price"])
            size = Decimal(change["size"])
            side = change.get("side", "").upper()

            book = self.books[asset_id]
            target = book["bids"] if side == "BUY" else book["asks"]

            if size == 0:
                target.pop(price, None)
            else:
                target[price] = size

        if self.on_price_change:
            self.on_price_change(data)

    def _handle_trade(self, data: dict):
        if self.on_trade:
            self.on_trade(data)

    def _handle_best_bid_ask(self, data: dict):
        if self.on_best_bid_ask:
            self.on_best_bid_ask(data)

    def _handle_tick_size_change(self, data: dict):
        if self.on_tick_size_change:
            self.on_tick_size_change(data)

    def _handle_market_resolved(self, data: dict):
        # Remove resolved assets from tracking
        asset_id = data.get("asset_id")
        if asset_id:
            self.books.pop(asset_id, None)
        if self.on_market_resolved:
            self.on_market_resolved(data)

    def _handle_new_market(self, data: dict):
        if self.on_new_market:
            self.on_new_market(data)
