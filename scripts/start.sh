#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs runtime
touch logs/trading.log logs/error.log logs/structured.jsonl

if [[ ! -x "venv/bin/python" ]]; then
  echo "Missing virtualenv interpreter at venv/bin/python" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec venv/bin/python -u main.py
