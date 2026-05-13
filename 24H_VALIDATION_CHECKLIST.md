# 24H Paper-Trading Validation Checklist

## Preconditions
- Confirm `.env` has `TRADING_MODE=paper`.
- Confirm `.env` has `TRADING_SYMBOLS=BTC/USDT,ETH/USDT`.
- Confirm leverage caps are conservative: `DEFAULT_LEVERAGE=1`, `MAX_LONG_LEVERAGE=3`, `MAX_SHORT_LEVERAGE=2`.
- Confirm the virtualenv exists at `venv/bin/python`.
- Confirm `future-trader.service` is installed and enabled.

## Start
1. Run `systemctl start future-trader.service`.
2. Confirm the service is active with `systemctl status future-trader.service`.
3. Confirm logs are being written to `logs/trading.log`, `logs/error.log`, and `logs/structured.jsonl`.
4. Confirm `runtime/health.json` appears within the first minute.

## Watch Criteria
- `scripts/status.sh` should report a recent `health.json` timestamp.
- Health age should stay under 120 seconds.
- `exchange_connected` should stay `true` in healthy runs.
- `open_positions` should remain within configured limits.
- Symbol universe should remain limited to BTC and ETH.

## Failure Signals
- `runtime/health.json` missing or stale for more than 120 seconds.
- `systemctl status` shows restart loops or a failed unit.
- `logs/error.log` contains repeated traceback output.
- Structured logs stop advancing while the service is still active.
- Any accidental `TRADING_MODE=live` setting appears in the environment.

## Recovery
1. Run `scripts/status.sh` to check whether the issue is health, connectivity, or a crash loop.
2. If the health file is stale, inspect `logs/error.log` and the systemd journal.
3. If the process is stuck, use `systemctl restart future-trader.service`.
4. If exchange connectivity is broken, stop the service and fix credentials or network before restarting.
5. If any live-trading setting is present, stop immediately and restore paper-mode configuration.

## End Of Run
- Capture `systemctl status future-trader.service`.
- Archive `logs/trading.log`, `logs/error.log`, `logs/structured.jsonl`, and `runtime/health.json`.
- Review filled trades, restart count, and the last health timestamp.