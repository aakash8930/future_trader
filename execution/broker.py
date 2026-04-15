# execution/broker.py
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import ccxt

from execution.position import Position
from config.live import BINANCE_LIVE_API


class PaperBroker:
    """
    Simulates order execution (paper trading).
    """

    def __init__(self):
        self.position: Optional[Position] = None

    def open_position(self, side: str, price: float, qty: float, symbol: str | None = None) -> Position:
        self.position = Position(
            side=side,
            entry_price=price,
            qty=qty,
            entry_time=datetime.utcnow(),
        )
        return self.position

    def add_to_position(self, price: float, qty: float):
        if not self.position:
            return
        self.position.add_to_position(price, qty)

    def close_position(self, price: float, symbol: str | None = None) -> float:
        if self.position is None:
            return 0.0

        pnl = self.position.pnl(price)
        self.position = None
        return pnl


class ShadowBroker(PaperBroker):
    """
    Shadow trading broker.
    Executes ZERO real orders.
    Mirrors live behavior for validation.
    """

    def __init__(self):
        super().__init__()
        self._persist_path = "logs/position_state.json"
        self._load_position()

    def _persist_position(self):
        if self.position is None:
            import os
            if os.path.exists(self._persist_path):
                os.remove(self._persist_path)
            return

        import json
        data = {
            "symbol": getattr(self, 'symbol', None),
            "side": self.position.side,
            "entry_price": self.position.entry_price,
            "avg_entry": self.position.avg_entry,
            "qty": self.position.qty,
            "entry_time": self.position.entry_time.isoformat(),
            "add_count": self.position.add_count,
            "stop_loss": getattr(self, "_cached_stop_loss", None),
            "take_profit": getattr(self, "_cached_take_profit", None),
        }
        with open(self._persist_path, "w") as f:
            json.dump(data, f)

    def update_levels(self, stop_loss: float = None, take_profit: float = None):
        """Update and persist SL/TP levels without changing position."""
        self._cached_stop_loss = stop_loss
        self._cached_take_profit = take_profit
        self._persist_position()

    def _load_position(self):
        import os, json
        if not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r") as f:
                data = json.load(f)
            # If SL/TP are null, the position is incomplete - clear it
            if data.get("stop_loss") is None or data.get("take_profit") is None:
                print(f"[SHADOW] Clearing stale position (missing SL/TP): {data.get('symbol')}")
                os.remove(self._persist_path)
                return
            self.position = Position(
                side=data["side"],
                entry_price=data["entry_price"],
                qty=data["qty"],
                entry_time=datetime.fromisoformat(data["entry_time"]),
            )
            self.position.avg_entry = data["avg_entry"]
            self.position.add_count = data["add_count"]
            self.symbol = data.get("symbol")
            self._cached_stop_loss = data.get("stop_loss")
            self._cached_take_profit = data.get("take_profit")
            print(f"[SHADOW] Restored position: {data.get('symbol')} SL={self._cached_stop_loss:.4f} TP={self._cached_take_profit:.4f}")
        except Exception:
            if os.path.exists(self._persist_path):
                os.remove(self._persist_path)

    def open_position(self, side: str, price: float, qty: float, symbol: str | None = None) -> Position:
        print(f"[SHADOW] OPEN {side} {symbol} qty={qty:.6f} @ {price:.2f}")
        pos = super().open_position(side, price, qty, symbol)
        self.symbol = symbol
        self._persist_position()
        return pos

    def close_position(self, price: float, symbol: str | None = None) -> float:
        pnl = super().close_position(price, symbol)
        self._persist_position()
        print(f"[SHADOW] CLOSE {symbol} pnl={pnl:.4f}")
        return pnl

    def add_to_position(self, price: float, qty: float):
        if not self.position:
            return
        self.position.add_to_position(price, qty)
        self._persist_position()


class LiveBroker:
    """
    Executes real market orders via ccxt (spot only, LONG only).
    Supports both Binance live trading and testnet (demo) mode.
    """

    POSITION_STATE_PATH = Path("logs/position_state.json")

    def __init__(
        self,
        exchange_name: str,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
    ):
        if exchange_name != "binance":
            raise ValueError(f"Unsupported exchange: {exchange_name}")

        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )

        # Force testnet API URLs and enable sandbox mode
        if testnet:
            self.exchange.urls["api"] = {
                "public": "https://testnet.binance.vision/api",
                "private": "https://testnet.binance.vision/api",
            }
            self.exchange.set_sandbox_mode(True)
            print(f"[BROKER] Using BINANCE TESTNET sandbox")
        else:
            self.exchange.urls["api"] = {
                "public": f"{BINANCE_LIVE_API}/api/v3",
                "private": f"{BINANCE_LIVE_API}/api/v3",
            }
            print(f"[BROKER] Using BINANCE LIVE")

        # Load markets unconditionally - required for amount_to_precision
        self.testnet = testnet
        self.position: Optional[Position] = None
        self._markets_loaded = False
        self.symbol: Optional[str] = None
        self._cached_stop_loss: Optional[float] = None
        self._cached_take_profit: Optional[float] = None

        # Ensure logs directory exists and resume any open position
        self.POSITION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._load_position()

        # Force load markets now
        self._ensure_markets_loaded()

        # Sync open positions from exchange to prevent duplicate trades on restart
        self._sync_from_exchange()

    def _ensure_markets_loaded(self):
        """Unconditionally load markets. Call this before any order operation."""
        if self._markets_loaded:
            return
        try:
            self.exchange.load_markets()
            self._markets_loaded = True
            print("[BROKER] markets loaded")
        except Exception as e:
            print(f"[BROKER] load_markets() failed ({type(e).__name__}): {e}")
            raise  # re-raise so caller knows

    def _persist_position(self):
        if self.position is None:
            if self.POSITION_STATE_PATH.exists():
                self.POSITION_STATE_PATH.unlink(missing_ok=True)
            return
        data = {
            "symbol": self.symbol,
            "side": self.position.side,
            "entry_price": self.position.entry_price,
            "avg_entry": self.position.avg_entry,
            "qty": self.position.qty,
            "entry_time": self.position.entry_time.isoformat(),
            "add_count": self.position.add_count,
            "stop_loss": self._cached_stop_loss,
            "take_profit": self._cached_take_profit,
        }
        self.POSITION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.POSITION_STATE_PATH, "w") as f:
            json.dump(data, f)

    def _load_position(self):
        """Load open position from logs/position_state.json on startup."""
        import json, os
        if not self.POSITION_STATE_PATH.exists():
            return
        try:
            with open(self.POSITION_STATE_PATH, "r") as f:
                data = json.load(f)
            # If SL/TP are null, the position is incomplete - clear it
            if data.get("stop_loss") is None or data.get("take_profit") is None:
                print(f"[BROKER] Clearing stale position (missing SL/TP): {data.get('symbol')}")
                self.POSITION_STATE_PATH.unlink(missing_ok=True)
                return
            self.position = Position(
                side=data["side"],
                entry_price=data["entry_price"],
                qty=data["qty"],
                entry_time=datetime.fromisoformat(data["entry_time"]),
            )
            self.position.avg_entry = data["avg_entry"]
            self.position.add_count = data["add_count"]
            self.symbol = data.get("symbol")
            self._cached_stop_loss = data.get("stop_loss")
            self._cached_take_profit = data.get("take_profit")
            print(f"[BROKER] Restored position: {data.get('symbol')} SL={self._cached_stop_loss:.4f} TP={self._cached_take_profit:.4f}")
        except Exception:
            if self.POSITION_STATE_PATH.exists():
                self.POSITION_STATE_PATH.unlink(missing_ok=True)

    def _sync_from_exchange(self):
        """Fetch open positions from exchange and restore into broker state. Prevents duplicate trades on restart."""
        try:
            self._ensure_markets_loaded()
            balance = self.exchange.fetch_balance()
            total_position_value = 0.0
            positions = []

            # Binance spot: check total portfolio value in USDT
            free_usdt = float(balance.get("free", {}).get("USDT", 0.0))
            total_usdt = float(balance.get("total", {}).get("USDT", 0.0))
            locked_usdt = total_usdt - free_usdt

            # Scan all assets for non-zero quantities (excluding USDT itself)
            for asset, balance_info in balance.items():
                if asset == "USDT" or not isinstance(balance_info, dict):
                    continue
                free = float(balance_info.get("free", 0.0) or 0.0)
                locked = float(balance_info.get("locked", 0.0) or 0.0)
                total = free + locked
                if total <= 0:
                    continue
                # Convert to USDT using current price if needed
                # For spot, any non-zero quantity in a asset means we hold it
                positions.append((asset, total))

            if not positions:
                print("[BROKER] No open positions found on exchange")
                return

            print(f"[BROKER] Exchange reports holdings: {positions}")
            print("[BROKER] Position sync: bot state loaded from logs only — exchange positions are live and independent")
        except Exception as e:
            print(f"[BROKER] Position sync failed: {e}")

    def update_levels(self, stop_loss: float = None, take_profit: float = None):
        """Update and persist SL/TP levels without changing position."""
        self._cached_stop_loss = stop_loss
        self._cached_take_profit = take_profit
        self._persist_position()

    def get_balance_usdt(self) -> float:
        self._ensure_markets_loaded()
        balance = self.exchange.fetch_balance()
        return float(
            balance.get("free", {}).get("USDT")
            or balance.get("total", {}).get("USDT")
            or 0.0
        )

    def _normalize_qty(self, symbol: str, qty: float) -> float:
        self._ensure_markets_loaded()
        qty = float(self.exchange.amount_to_precision(symbol, qty))
        min_amount = self.exchange.market(symbol).get("limits", {}).get("amount", {}).get("min")

        if min_amount and qty < float(min_amount):
            raise ValueError(f"Order qty too small for {symbol}")

        return qty

    def _validate_notional(self, symbol: str, qty: float, price: float) -> None:
        min_cost = self.exchange.market(symbol).get("limits", {}).get("cost", {}).get("min")
        if min_cost and qty * price < float(min_cost):
            raise ValueError(f"Order notional too small for {symbol}")

    def open_position(self, side: str, price: float, qty: float, symbol: str) -> Position:
        self._ensure_markets_loaded()
        if self.position:
            raise RuntimeError("Position already open")

        if side != "LONG":
            raise ValueError("Live spot broker supports LONG only")

        qty = self._normalize_qty(symbol, qty)
        self._validate_notional(symbol, qty, price)

        order = self.exchange.create_order(symbol, "market", "buy", qty)
        fill_price = float(order.get("average") or price)
        filled_qty = float(order.get("filled") or qty)

        if filled_qty <= 0:
            raise RuntimeError("Buy order not filled")

        self.position = Position(
            side=side,
            entry_price=fill_price,
            qty=filled_qty,
            entry_time=datetime.utcnow(),
        )
        self.symbol = symbol
        self._persist_position()
        return self.position

    def close_position(self, price: float, symbol: str) -> float:
        if not self.position:
            return 0.0

        qty = self._normalize_qty(symbol, self.position.qty)
        order = self.exchange.create_order(symbol, "market", "sell", qty)
        exit_price = float(order.get("average") or price)

        pnl = self.position.pnl(exit_price)
        self.position = None
        self._persist_position()
        return pnl