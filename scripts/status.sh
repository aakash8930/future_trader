#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="future-trader.service"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEALTH_FILE="$ROOT_DIR/runtime/health.json"
PYTHON_BIN="$ROOT_DIR/venv/bin/python"

echo "=== Service Status ==="
systemctl status "$SERVICE_NAME" --no-pager || true

echo
echo "=== Main PID / Uptime / Memory ==="
MAIN_PID="$(systemctl show "$SERVICE_NAME" -p MainPID --value 2>/dev/null || echo 0)"
if [[ "$MAIN_PID" != "0" ]] && [[ -d "/proc/${MAIN_PID}" ]]; then
  ps -p "$MAIN_PID" -o pid=,etime=,%mem=,rss=,state=,cmd=
else
  echo "Service process is not running."
fi

echo
echo "=== Active Process State ==="
systemctl show "$SERVICE_NAME" -p ActiveState,SubState,Result --no-pager || true

echo
echo "=== Latest Trading Logs ==="
tail -n 30 logs/trading.log 2>/dev/null || echo "logs/trading.log not found"

echo
echo "=== Latest Error Logs ==="
tail -n 30 logs/error.log 2>/dev/null || echo "logs/error.log not found"

echo
echo "=== Health Check ==="
if [[ -f "$HEALTH_FILE" ]]; then
  if [[ -x "$PYTHON_BIN" ]]; then
    "$PYTHON_BIN" - <<'PY' "$HEALTH_FILE"
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
updated_at = data.get("updated_at")
if not updated_at:
    print("health.json missing updated_at")
    raise SystemExit(1)

stamp = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
age = (datetime.now(timezone.utc) - stamp).total_seconds()
print(f"health status: {data.get('status')}")
print(f"health age: {age:.1f}s")
print(f"health balance: {data.get('balance')}")
print(f"health open_positions: {data.get('open_positions')}")
print(f"health exchange_connected: {data.get('exchange_connected')}")
if age > 120:
    print("WARNING: health.json is stale (>120s)")
PY
  else
    echo "WARNING: venv/bin/python not found; cannot parse health.json"
  fi
else
  echo "WARNING: runtime/health.json not found"
fi
