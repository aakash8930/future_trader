#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="future-trader.service"

if systemctl list-unit-files --type=service | grep -q "^${SERVICE_NAME}"; then
  systemctl restart "$SERVICE_NAME"
  echo "Restarted ${SERVICE_NAME}"
else
  echo "Service ${SERVICE_NAME} is not installed."
fi
