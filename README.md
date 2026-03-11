# AI Crypto Trading System (RUDRA-ALPHA)

## Overview
An AI-driven, fully automated crypto trading system designed for
robustness, risk control, and real-world exchange conditions.

## Key Features
- Machine-learning directional models
- Regime-aware ensemble inference
- Multi-symbol dynamic selection
- Professional risk management
- Drawdown kill-switch
- Consecutive loss breaker
- Shadow trading mode (capital-free validation)

## Trading Modes
- paper   – simulated execution
- shadow  – real-time execution without capital
- live    – real trading (locked by default)

## Current Status
✅ Validated via extended shadow trading  
✅ All risk safeguards active  
❌ Live trading disabled until funded

## Safety Mechanisms
- Daily & weekly drawdown limits
- Max active positions
- Cooldowns
- Model quality gates
- Automatic symbol disabling

**Symbol configuration:**
You can specify `TRADING_SYMBOLS` in your environment as a
comma-separated list.  To temporarily skip particular markets, set
`EXCLUDED_SYMBOLS` to another comma-separated list; excluded tickers are
removed from the `TRADING_SYMBOLS` list at startup without any code
changes.

## Disclaimer
This system is provided for educational and research purposes.
Live trading is disabled by default and requires explicit unlock.