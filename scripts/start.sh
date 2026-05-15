#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs runtime
# Create missing files only — do not update timestamps or ownership on existing files
for f in logs/trading.log logs/error.log logs/structured.jsonl; do
  if [ ! -e "$f" ]; then
    mkdir -p "$(dirname "$f")"
    : > "$f"
  fi
done

if [[ ! -x "venv/bin/python" ]]; then
  echo "Missing virtualenv interpreter at venv/bin/python" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec venv/bin/python -u main.py
