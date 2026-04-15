import time
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse

from . import position, events
from logs.trade_store import (
    get_performance_summary,
    get_trades,
    get_equity_curve,
    get_distinct_symbols,
    get_latest_balance,
)
from logs import trade_store
from .schema import (
    ApiResponse, HealthResponse, SystemStatus, PerformanceStats,
    EquityPoint, TradeRow, TradeEvent, PositionState
)
from config.shared_state import get_shared_state, update_shared_state
from config.live import LIVE_UNLOCK_TOKEN


_START_TIME = time.time()
_MODE = os.getenv("TRADING_MODE", "shadow").strip().lower()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure logs dir exists
    Path("logs").mkdir(exist_ok=True)

    # Auto-start the incremental trainer so models continuously train on live data
    try:
        from train.incremental_trainer import get_trainer
        trainer = get_trainer()
        print(f"[DASHBOARD] Incremental trainer started (interval={trainer.retrain_interval_seconds}s)")
    except Exception as e:
        print(f"[DASHBOARD] Could not start incremental trainer: {e}")

    yield


app = FastAPI(title="Trading Dashboard API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path), html=True), name="static")


def _ok(data):
    return {"ok": True, "data": data, "timestamp": datetime.utcnow().isoformat()}


def _err(msg):
    return {"ok": False, "error": msg, "timestamp": datetime.utcnow().isoformat()}


@app.get("/")
async def root():
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        from starlette.responses import FileResponse
        return FileResponse(str(index_path))
    return {"message": "Trading Dashboard API", "docs": "/docs"}


@app.get("/api/v1/health")
async def health():
    return _ok(HealthResponse(
        status="ok",
        mode=_MODE,
        uptime_seconds=round(time.time() - _START_TIME, 1),
    ))


@app.get("/api/v1/status")
async def status():
    stats = get_performance_summary()
    positions = position.get_active_positions()
    balance = get_latest_balance()

    return _ok(SystemStatus(
        mode=_MODE,
        symbols=get_distinct_symbols(),
        balance=balance,
        active_positions=[PositionState(**p) for p in positions],
        starting_balance=float(os.getenv("PAPER_STARTING_BALANCE_USDT", "1000")),
    ))


@app.get("/api/v1/positions")
async def positions():
    return _ok(position.get_active_positions())


@app.get("/api/v1/stats")
async def stats():
    return _ok(get_performance_summary())


@app.get("/api/v1/equity-curve")
async def equity_curve(limit: int = Query(500, ge=1, le=2000)):
    return _ok(get_equity_curve(limit=limit))


@app.get("/api/v1/trades")
async def trades(
    symbol: str = Query(None),
    side: str = Query(None),
    exit_reason: str = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    return _ok(get_trades(
        symbol=symbol, side=side, exit_reason=exit_reason,
        limit=limit, offset=offset,
    ))


@app.get("/api/v1/symbols")
async def symbols():
    return _ok(get_distinct_symbols())


@app.get("/api/v1/events/stream")
async def events_stream():
    return await events.stream_events(None)


# Control endpoints
_control_state = {
    "paused": False,
    "mode": os.getenv("TRADING_MODE", "shadow").strip().lower(),
}


@app.get("/api/v1/control/state")
async def control_state():
    return _ok(_control_state)


@app.post("/api/v1/control/pause")
async def control_pause():
    _control_state["paused"] = True
    return _ok({"message": "Trading paused", "paused": True})


@app.post("/api/v1/control/resume")
async def control_resume():
    _control_state["paused"] = False
    return _ok({"message": "Trading resumed", "paused": False})


@app.post("/api/v1/control/mode")
async def control_mode(mode: str = Query(...)):
    if mode not in {"paper", "shadow", "live", "demo"}:
        return _err(f"Invalid mode: {mode}")
    _control_state["mode"] = mode
    update_shared_state(mode=mode)
    return _ok({"message": f"Mode set to {mode}", "mode": mode})


@app.post("/api/v1/control/timeframe")
async def control_timeframe(timeframe: str = Query(...)):
    valid_timeframes = {"1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"}
    if timeframe not in valid_timeframes:
        return _err(f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}")
    update_shared_state(timeframe=timeframe)
    return _ok({"timeframe": timeframe, "message": f"Timeframe set to {timeframe}"})


@app.post("/api/v1/control/live-unlock")
async def control_live_unlock(token: str = Query(...)):
    if token == LIVE_UNLOCK_TOKEN:
        update_shared_state(live_unlock=True)
        return _ok({"message": "Live trading unlocked", "live_unlock": True})
    return _err("Invalid unlock token")


@app.post("/api/v1/control/retrain")
async def control_retrain(on_demand: bool = Query(True)):
    from train.incremental_trainer import get_trainer
    trainer = get_trainer()
    if on_demand:
        trainer.trigger_retrain()
    return _ok({"message": "Retrain triggered", "on_demand": on_demand})


@app.post("/api/v1/control/retrain/stop")
async def control_retrain_stop():
    from train.incremental_trainer import stop_trainer
    stop_trainer()
    return _ok({"message": "Background trainer stopped"})


@app.get("/api/v1/control/retrain/status")
async def control_retrain_status():
    state = get_shared_state()
    from train.incremental_trainer import get_trainer_status
    trainer_info = get_trainer_status()
    return _ok({
        "retrain_interval_seconds": state.retrain_interval_seconds,
        "last_retrain_timestamp": state.last_retrain_timestamp,
        "trainer_active": trainer_info.get("active", False),
        "trainer_symbols": trainer_info.get("symbols", []),
    })


@app.get("/api/v1/control/config")
async def control_config():
    state = get_shared_state()
    return _ok({
        "mode": state.mode,
        "timeframe": state.timeframe,
        "paused": state.paused,
        "live_unlock": state.live_unlock,
        "retrain_interval_seconds": state.retrain_interval_seconds,
        "last_retrain_timestamp": state.last_retrain_timestamp,
        "risk_per_trade": state.risk_per_trade,
        "max_active_positions": state.max_active_positions,
        "strategy_min_prob": state.strategy_min_prob,
        "cooldown_minutes": state.cooldown_minutes,
    })


@app.post("/api/v1/control/retrain/interval")
async def control_retrain_interval(seconds: int = Query(..., ge=300, le=86400)):
    update_shared_state(retrain_interval_seconds=seconds)
    # Update running trainer interval
    from train.incremental_trainer import update_retrain_interval
    update_retrain_interval(seconds)
    return _ok({"retrain_interval_seconds": seconds, "message": f"Retrain interval set to {seconds}s"})


@app.post("/api/v1/control/risk")
async def control_risk(risk: float = Query(..., ge=0.005, le=0.10)):
    """Set risk per trade (fraction, e.g. 0.01 = 1%)."""
    update_shared_state(risk_per_trade=risk)
    return _ok({"risk_per_trade": risk, "message": f"Risk per trade set to {risk * 100:.1f}%"})


@app.post("/api/v1/control/max-positions")
async def control_max_positions(positions: int = Query(..., ge=1, le=10)):
    """Set max active positions."""
    update_shared_state(max_active_positions=positions)
    return _ok({"max_active_positions": positions, "message": f"Max positions set to {positions}"})


@app.post("/api/v1/control/min-prob")
async def control_min_prob(prob: float = Query(..., ge=0.40, le=0.70)):
    """Set min model probability threshold (fraction)."""
    update_shared_state(strategy_min_prob=prob)
    return _ok({"strategy_min_prob": prob, "message": f"Min probability set to {prob * 100:.0f}%"})


@app.post("/api/v1/control/cooldown")
async def control_cooldown(minutes: int = Query(..., ge=5, le=120)):
    """Set entry cooldown in minutes."""
    update_shared_state(cooldown_minutes=minutes)
    return _ok({"cooldown_minutes": minutes, "message": f"Cooldown set to {minutes}m"})


@app.get("/api/v1/price-history")
async def price_history(
    symbol: str = Query("BTC/USDT"),
    timeframe: str = Query("15m"),
    limit: int = Query(200, ge=20, le=1000),
):
    """
    Fetch live OHLCV from Binance for the live price chart.
    Returns candlestick data with timestamp, open, high, low, close, volume.
    """
    try:
        from data.fetcher import MarketDataFetcher
        fetcher = MarketDataFetcher(
            exchange_name="binance",
            fallback_exchanges=["bybit", "kraken", "okx"],
            timeout_ms=15000,
        )
        df = fetcher.fetch_ohlcv(symbol, timeframe, limit=limit)
        if df is None or df.empty:
            return _err(f"Symbol {symbol} not supported on Binance")

        # Use closed candles only
        df = df.iloc[:-1].copy()

        candles = []
        for _, row in df.iterrows():
            candles.append({
                "time": int(row["time"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })

        return _ok({
            "symbol": symbol,
            "timeframe": timeframe,
            "count": len(candles),
            "candles": candles,
        })
    except Exception as e:
        return _err(str(e)[:200])


@app.get("/api/v1/data-source")
async def data_source():
    """
    Returns the current data source status for the dashboard indicator.
    live = Binance real-time feed; sql = PostgreSQL cached data.
    """
    db_url = os.getenv("DATABASE_PUBLIC_URL") or os.getenv("DATABASE_URL")
    using_sql = bool(db_url)
    return _ok({
        "source": "live" if not using_sql else "sql",
        "label": "LIVE BINANCE" if not using_sql else "POSTGRESQL",
        "description": "Real-time Binance market data" if not using_sql else "PostgreSQL trade history",
    })


@app.get("/api/v1/signals")
async def signals():
    """
    Returns current signal data for all watched symbols:
    - live indicator values (ADX, RSI, ATR%, price vs EMA200, EMA cross)
    - model probability and threshold
    - coin selector score and reason for rejection/acceptance
    """
    try:
        from data.fetcher import MarketDataFetcher
        from features.technicals import compute_core_features
        from execution.coin_selector import CoinSelector, _has_trained_model
        from models.direction import DirectionModel

        # Read timeframe from shared state
        state = get_shared_state()
        tf = state.timeframe or "15m"

        fetcher = MarketDataFetcher(exchange_name="binance", fallback_exchanges=["bybit", "kraken", "okx"], timeout_ms=15000)
        selector = CoinSelector(timeframe=tf, lookback=240)

        symbols = [s for s in selector.DEFAULT_SYMBOLS if fetcher.is_symbol_supported(s)]
        results = []

        for symbol in symbols:
            try:
                df = fetcher.fetch_ohlcv(symbol, tf, limit=250)
                df = df.iloc[:-1].copy()
                if len(df) < selector.lookback:
                    continue
                df = compute_core_features(df)
                df = df.dropna()
                if df.empty:
                    continue

                row = df.iloc[-1]
                price = float(row["close"])
                ema200 = float(row["ema200"])
                ema_fast = float(row["ema_fast"])
                ema_slow = float(row["ema_slow"])
                adx = float(row["adx"])
                atr_pct = float(row["atr_pct"])
                rsi = float(row["rsi"])
                dist_ema200 = float(row["dist_ema200"])
                above_ema200 = price > ema200
                bullish_cross = ema_fast > ema_slow

                # Model probability
                has_model = _has_trained_model(symbol)
                prob = 0.5
                threshold = 0.50
                if has_model:
                    try:
                        model = DirectionModel.for_symbol(symbol)
                        prob = float(model.predict_proba(df))
                        threshold = float(getattr(model, "long_threshold", 0.50))
                    except UnicodeEncodeError as e:
                        # Windows cp1252 encoding issue with certain characters
                        results.append({
                            "symbol": symbol,
                            "has_model": has_model,
                            "prob": 0.5,
                            "threshold": 0.50,
                            "score": 0.0,
                            "status": "ERROR",
                            "reason": f"Encoding error: {str(e)[:50]}",
                            "indicators": {}
                        })
                        continue
                    except Exception as e:
                        results.append({
                            "symbol": symbol,
                            "has_model": has_model,
                            "prob": 0.5,
                            "threshold": 0.50,
                            "score": 0.0,
                            "status": "ERROR",
                            "reason": str(e)[:60],
                            "indicators": {}
                        })

                # Coin selector scoring
                score_val = selector._score_symbol(symbol)
                adaptive_th = max(0.48, threshold - 0.01)
                strong_trend = above_ema200 and bullish_cross and adx >= 26.0 and atr_pct >= selector.min_atr_pct
                momentum_override = (
                    bullish_cross and adx >= 30.0 and atr_pct >= selector.min_atr_pct
                    and prob >= adaptive_th + 0.01 and dist_ema200 > -0.020
                )
                reasons = []
                if atr_pct < selector.min_atr_pct:
                    print(f"[ATR DEBUG] atr_pct={atr_pct:.6f} < min_atr_pct={selector.min_atr_pct:.6f} -> blocked")
                    reasons.append(f"atr_pct_low({atr_pct:.4f}<{selector.min_atr_pct:.4f})")
                vol_series = df["volume"]
                vol_ma = float(vol_series.rolling(20).mean().iloc[-1])
                cur_vol = float(row["volume"])
                vol_ratio = cur_vol / vol_ma if (np.isfinite(vol_ma) and vol_ma > 0 and np.isfinite(cur_vol)) else 1.0
                if vol_ratio < selector.soft_min_volume_ratio * 0.60:
                    reasons.append("volume_too_low")
                if not above_ema200 and dist_ema200 <= -0.035 and not momentum_override:
                    reasons.append("far_below_ema200")
                if rsi < 28.0:
                    reasons.append("rsi_oversold")
                elif rsi < selector.rsi_long_min:
                    reasons.append("rsi_below_thresh")
                elif rsi > selector.rsi_long_max:
                    reasons.append("rsi_overbought")
                if prob < adaptive_th - 0.035:
                    reasons.append("prob_below_thresh")

                if score_val is not None and score_val > -100 and not reasons:
                    status = "READY"
                    reason_str = ""
                elif score_val is not None and score_val > -100:
                    status = "SOFT_FAIL"
                    reason_str = ", ".join(reasons) if reasons else "marginal"
                else:
                    status = "SKIPPED"
                    reason_str = ", ".join(reasons) if reasons else "hard_blocked"

                results.append({
                    "symbol": symbol,
                    "has_model": has_model,
                    "prob": prob,
                    "threshold": threshold,
                    "score": score_val if score_val is not None else 0.0,
                    "status": status,
                    "reason": reason_str,
                    "indicators": {
                        "adx": adx if np.isfinite(adx) else 0.0,
                        "rsi": rsi if np.isfinite(rsi) else 50.0,
                        "atr_pct": atr_pct if np.isfinite(atr_pct) else 0.0,
                        "above_ema200": above_ema200,
                        "bullish_cross": bullish_cross,
                        "dist_ema200": dist_ema200 if np.isfinite(dist_ema200) else 0.0,
                        "regime": "trend_strong" if adx >= 28 and atr_pct >= 0.003 else "trend_weak" if adx >= 16 else "sideways",
                    }
                })
            except Exception as e:
                results.append({
                    "symbol": symbol,
                    "has_model": False,
                    "prob": 0.5,
                    "threshold": 0.50,
                    "score": 0.0,
                    "status": "ERROR",
                    "reason": str(e)[:60],
                    "indicators": {}
                })

        return _ok({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "timeframe": tf,
            "signals": results,
        })

    except Exception as e:
        return _err(str(e)[:200])


@app.get("/dashboard")
async def dashboard():
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        from starlette.responses import FileResponse
        return FileResponse(str(index_path))
    return {"error": "index.html not found"}
