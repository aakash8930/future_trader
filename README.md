# ALPHASEEKER — AI Crypto Trading System

A fully autonomous, AI-driven cryptocurrency trading system with regime-aware ML models, multi-symbol portfolio management, and real-time dashboard monitoring.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      main.py                              │
│              (entry point, dispatch)                     │
└──────────────────┬────────────────────────────────────────┘
                   │
     ┌─────────────┴──────────────┐
     │                              │
┌────▼──────────┐         ┌────────▼──────────┐
│ TradingRunner │         │MultiSymbolTrading │
│ (single sym)  │         │    System          │
└────┬──────────┘         └─────────┬──────────┘
     │                               │
     │  ┌─────────────────────────────┼─────────────┐
     │  │                             │             │
     │  ▼                             ▼             ▼
     │ Runner loop               Universe      CoinSelector
     │  • fetch OHLCV            Manager       (symbol scoring)
     │  • compute features       (top-k        + filtering
     │  • regime detection         selection)
     │  • signal gen
     │  • position mgmt
     │
     │  ┌──────────────────────────────────────┐
     └──►  Risk Layer                          │
     │  ├── MarketGuard  (daily drawdown kill) │
     │  ├── RiskState    (balance tracking)     │
     │  ├── AISupervisor (adaptive risk mult)  │
     │  ├── RegimeController (trend/ sideways) │
     │  └── Position sizing (fixed fractional)│
     │  └──────────────────────────────────────┘
     │
     └──►  Execution Layer
          ├── ShadowBroker  (paper / shadow mode)
          ├── LiveBroker    (real orders via CCXT)
          └── TradeLogger   (PostgreSQL trade record)
                      │
                      ▼
              ┌───────────────┐
              │  Dashboard    │
              │  (FastAPI)    │◄─── HTML/JS frontend
              │  port 8000    │
              └───────────────┘
```

---

## Features

### AI & Machine Learning
- **Directional ML Model** — 3-layer MLP (32→16→1) trained on triple-barrier labels
  (TP/SL hit within horizon)
- **Regime-Aware Ensemble** — symbol model + BTC-context model, weighted by detected
  market regime (trend_strong / trend_weak / sideways)
- **13 Technical Features** — EMA spread, RSI, RSI delta, ADX, ATR%, volume z-score,
  breakout strength, return, volatility, EMA200 distance, EMA fast slope

### Trading & Risk Management
- **Fixed-Fractional Position Sizing** — risk-controlled lot sizing based on balance
- **Trailing Stop Loss** — ATR-multiple trailing stop that locks in profit at TP1
- **Daily Drawdown Kill-Switch** — `MarketGuard` halts trading if daily loss exceeds limit
- **Consecutive Loss Breaker** — blocks new entries after N consecutive losses
- **Adaptive Risk Multiplier** — `AISupervisor` adjusts risk (0 / 0.5 / 1.0 / 1.25×)
  based on rolling win rate and drawdown
- **Cooldown Between Trades** — prevents over-trading per symbol
- **Strategy Edge Filter** — skips trades where expected edge < minimum threshold

### Symbol Selection (Multi-Symbol Mode)
- **Dynamic Universe** — CoinSelector scores symbols by probability × ADX × ATR% ×
  RSI health × volume ratio, with EMA200 trend filter
- **Top-K Selection** — picks top `max_active_positions × selector_top_k_multiplier`
  symbols from the scored universe
- **Model Quality Gate** — rejects symbols whose val_f1 / val_precision / val_recall
  fall below configurable thresholds

### Execution Modes
| Mode | Orders | Description |
|------|--------|-------------|
| `paper` | Simulated | Paper trading with fake fills |
| `shadow` | Real exchange, zero qty | Real-time signal validation without capital |
| `live` | Real market orders | Real trading — requires `LIVE_UNLOCK_TOKEN` |

### Operations
- **Graceful Shutdown** — SIGINT/SIGTERM handlers persist open positions before exit
- **Exchange Fallback** — auto-fails over to backup exchanges if primary is geo-blocked
- **Error Tracing** — full traceback logged on any run-loop exception
- **PostgreSQL Trade Logger** — every trade recorded with full metadata
- **Dashboard API** — FastAPI backend serving equity curve, trade history, positions,
  and system status; supports SSE event streaming

---

## Project Structure

```
.
├── main.py                        # Entry point
├── run.py                         # Simple wrapper
├── .env                           # All configuration (not committed)
├── requirements.txt               # Python dependencies
│
├── config/
│   ├── env_loader.py              # Loads .env into os.environ
│   └── live.py                    # LiveSettings dataclass + validation
│
├── execution/
│   ├── runner.py                  # Single-symbol trading loop
│   ├── multi_runner.py            # Multi-symbol orchestration
│   ├── broker.py                  # PaperBroker / ShadowBroker / LiveBroker
│   ├── shadow_broker.py           # (standalone, mirrored in broker.py)
│   ├── position.py                # Position dataclass (avg_entry, pyramid)
│   ├── strategy.py                # Signal generation + SL/TP calculation
│   ├── regime_controller.py       # Market regime detection + risk multiplier
│   ├── ai_supervisor.py           # Adaptive risk based on win rate
│   ├── market_guard.py            # Daily drawdown + consecutive loss kill
│   ├── coin_selector.py           # Symbol scoring + selection
│   └── universe_manager.py        # Symbol universe refresh logic
│
├── models/
│   ├── direction.py               # PyTorch MLP wrapper + inference
│   ├── ensemble.py                # Regime-weighted multi-model ensemble
│   └── <symbol>/                  # Per-symbol trained artifacts
│       ├── model.pt               # Trained torch state dict
│       ├── scaler.save            # sklearn StandardScaler
│       └── metadata.json          # Metrics, threshold, feature list
│
├── train/
│   └── train_direction_model.py   # Full training pipeline
│
├── features/
│   └── technicals.py              # compute_core_features() — all indicators
│
├── risk/
│   ├── sizing.py                  # fixed_fractional_size()
│   ├── limits.py                  # RiskLimits + RiskState
│   └── portfolio.py               # PortfolioGuard (exposure)
│
├── backtest/
│   ├── engine.py                  # Basic backtest engine
│   ├── simulator.py               # HistoricalSimulator (mirrors runner logic)
│   ├── vector_engine.py           # Fast vectorized backtest
│   ├── optimize_threshold.py       # Threshold optimization script
│   └── run*.py                    # Various backtest runners
│
├── logs/
│   ├── logger.py                  # TradeLogger (PostgreSQL)
│   └── event_logger.py            # EventLogger (JSONL, SSE streaming)
│
├── metrics/
│   ├── equity.py                  # Equity curve loading
│   ├── performance.py             # Performance summary
│   └── self_report.py             # DailyAIReport (CSV)
│
├── dashboard/
│   ├── app.py                     # FastAPI app
│   ├── run.py                     # uvicorn runner
│   ├── index.html                 # Dashboard frontend
│   ├── db.py                      # Dashboard DB queries (PostgreSQL)
│   ├── events.py                  # SSE event streaming
│   ├── position.py                # Active position queries
│   └── requirements.txt           # Dashboard dependencies
│
└── stats/
    ├── performance_report.py       # Full trade report from DB
    └── trade_stats.py             # Simple win/loss counter
```

---

## Setup

### 1. Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env   # or create from scratch
```

**Required variables:**

```env
# ── Trading ────────────────────────────────────────────────
TRADING_MODE=shadow          # paper | shadow | live
TRADING_SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT,AVAX/USDT,LINK/USDT
TRADING_TIMEFRAME=5m         # 1m | 3m | 5m | 15m | 30m | 1h | 4h | 1d
LOOKBACK_BARS=300

# ── Risk & Sizing ─────────────────────────────────────────
PAPER_STARTING_BALANCE_USDT=100
RISK_PER_TRADE=0.05          # 5% of balance per trade
ENTRY_COOLDOWN_MINUTES=30
MAX_ACTIVE_POSITIONS=2

# ── Strategy Thresholds ────────────────────────────────────
STRATEGY_MIN_ADX=12
STRATEGY_MIN_ATR_PCT=0.0003
STRATEGY_MIN_EXPECTED_EDGE=0.00005
STRATEGY_STOP_ATR_MULT=1.45
STRATEGY_TAKE_ATR_MULT=3.2
STRATEGY_TRAIL_ATR_MULT=0.9
STRATEGY_RSI_LONG_MIN=40
STRATEGY_RSI_LONG_MAX=70

# ── Symbol Selection ───────────────────────────────────────
UNIVERSE_REFRESH_MINUTES=60
SELECTOR_TOP_K_MULTIPLIER=2
SELECTOR_MIN_ATR_PCT=0.0010
SELECTOR_SOFT_MIN_VOLUME_RATIO=0.15
REQUIRE_MODEL_QUALITY=true
MIN_MODEL_VAL_F1=0.10
MIN_MODEL_VAL_PRECISION=0.10
MIN_MODEL_VAL_RECALL=0.10

# ── Exchange ──────────────────────────────────────────────
EXCHANGE_NAME=binance
EXCHANGE_FALLBACKS=bybit,kraken,okx
EXCHANGE_TIMEOUT_MS=20000

# ── Loop ──────────────────────────────────────────────────
LOOP_SLEEP_SECONDS=20

# ── Live Trading (only if TRADING_MODE=live) ──────────────
EXCHANGE_API_KEY=your_binance_api_key
EXCHANGE_API_SECRET=your_binance_api_secret
EXCHANGE_TESTNET=true         # false for mainnet
LIVE_UNLOCK_TOKEN=YES_I_UNDERSTAND

# ── Database ──────────────────────────────────────────────
DATABASE_URL=postgresql://user:pass@host/dbname
```

> **Note:** `.env` is gitignored — never commit your secrets.

### 2. Install Dependencies

```bash
# Backend dependencies
pip install -r requirements.txt

# Dashboard dependencies (separate venv recommended)
cd dashboard
pip install -r requirements.txt
cd ..
```

### 3. Train Models (optional — pre-trained models are included)

```bash
# Override training params via env vars (all optional)
TRAIN_EPOCHS=20 TRAIN_BATCH_SIZE=128 python -m train.train_direction_model
```

### 4. Start the Backend (Trading Bot)

```bash
# Single symbol
python main.py

# Or directly:
python run.py
```

Logs output directly to stdout. Position state is persisted to `logs/position_state.json`.

### 5. Start the Dashboard

```bash
cd dashboard
python run.py
# API available at http://localhost:8000
# Dashboard at       http://localhost:8000/dashboard
```

### 6. Start Both (recommended)

In two separate terminals:

```bash
# Terminal 1 — trading bot
python main.py

# Terminal 2 — dashboard (run from project root)
python -m dashboard.run
```

---

## Dashboard

Access at `http://localhost:8000/dashboard`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/status` | GET | System status, balance, active positions |
| `/api/v1/stats` | GET | Aggregate PnL, win rate, expectancy, drawdown |
| `/api/v1/positions` | GET | Currently open positions |
| `/api/v1/trades` | GET | Trade history (filterable by symbol/side/exit_reason) |
| `/api/v1/equity-curve` | GET | Equity curve data points |
| `/api/v1/events/stream` | GET | SSE stream of real-time trade/skip events |
| `/api/v1/control/pause` | POST | Pause trading |
| `/api/v1/control/resume` | POST | Resume trading |

---

## Backtesting

```bash
# Historical simulation (mirrors live runner logic exactly)
python -m backtest.run_historical

# Fast vectorized backtest
python -m backtest.run_vector

# Walk-forward analysis
python -m backtest.run_walkforward

# Threshold optimization
python -m backtest.run_optimize_threshold
```

Results are saved to `data_outputs/`.

---

## Directory Summary

| Directory | Purpose |
|-----------|---------|
| `execution/` | Live trading loop, order execution, risk management |
| `models/` | Trained ML models and ensemble logic |
| `train/` | Model training pipeline |
| `features/` | Technical indicator computation |
| `risk/` | Position sizing, drawdown limits, portfolio guard |
| `backtest/` | Historical simulation and backtesting tools |
| `logs/` | Trade logger (PostgreSQL) and event logger (JSONL) |
| `dashboard/` | FastAPI web dashboard with HTML frontend |
| `metrics/` | Performance analytics and equity tracking |
| `stats/` | Trade statistics reporting |

---

## Safety Features

| Feature | File | What it does |
|---------|------|--------------|
| Daily drawdown kill | `execution/market_guard.py` | Halts all trading if daily loss exceeds threshold |
| Consecutive loss breaker | `execution/market_guard.py` | Blocks new entries after N consecutive losses |
| Adaptive risk | `execution/ai_supervisor.py` | Reduces risk multiplier when win rate drops |
| Regime filter | `execution/regime_controller.py` | Reduces/triggers risk by detected regime |
| Position size cap | `risk/sizing.py` | Prevents over-sized positions |
| Notional minimum check | `execution/broker.py` | LiveBroker validates order notional > exchange min |
| LIVE_UNLOCK_TOKEN | `config/live.py` | Live trading requires explicit opt-in |
| Graceful shutdown | `execution/runner.py` | SIGINT/SIGTERM persist position before exit |

---

## Configuration Reference

All configuration is via environment variables (`.env`). Key knobs:

### Trading
| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_MODE` | `shadow` | `paper` \| `shadow` \| `live` |
| `TRADING_SYMBOLS` | `BTC/USDT` | Comma-separated symbol list |
| `TRADING_TIMEFRAME` | `5m` | Candle timeframe |
| `LOOKBACK_BARS` | `300` | Historical bars for feature computation |

### Risk
| Variable | Default | Description |
|----------|---------|-------------|
| `RISK_PER_TRADE` | `0.05` | Fraction of balance risked per trade |
| `ENTRY_COOLDOWN_MINUTES` | `30` | Minimum minutes between entries (same symbol) |
| `MAX_ACTIVE_POSITIONS` | `2` | Maximum concurrent open positions |

### Model Quality Gates
| Variable | Default | Description |
|----------|---------|-------------|
| `REQUIRE_MODEL_QUALITY` | `false` | Enable/disable model quality filtering |
| `MIN_MODEL_VAL_F1` | `0.10` | Minimum validation F1 score |
| `MIN_MODEL_VAL_PRECISION` | `0.10` | Minimum validation precision |
| `MIN_MODEL_VAL_RECALL` | `0.10` | Minimum validation recall |

### Training (optional overrides)
| Variable | Default | Description |
|----------|---------|-------------|
| `TRAIN_EPOCHS` | `10` | Training epochs |
| `TRAIN_BATCH_SIZE` | `256` | Batch size |
| `TRAIN_LR` | `1e-3` | Learning rate |
| `TRAIN_PATIENCE` | `3` | Early stopping patience |
| `TRAIN_HORIZON` | `12` | Triple-barrier look-ahead (bars) |
| `TRAIN_STOP_ATR_MULT` | `2.0` | Stop loss ATR multiple |
| `TRAIN_TAKE_ATR_MULT` | `3.0` | Take profit ATR multiple |
#   t r a d e r  
 