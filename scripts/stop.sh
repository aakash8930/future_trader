#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="future-trader.service"

if systemctl list-unit-files --type=service | grep -q "^${SERVICE_NAME}"; then
  systemctl stop "$SERVICE_NAME"
  echo "Stopped ${SERVICE_NAME}"
else
  echo "Service ${SERVICE_NAME} is not installed."
fi
