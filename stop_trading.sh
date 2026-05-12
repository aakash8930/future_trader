#!/bin/bash
# Script to stop the trading system gracefully

# Set the working directory
cd /home/aakash/Trading/future_trading/future_trader

# Get the PID file
PID_FILE="logs/trading_system.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Stopping trading system with PID $PID..."

    # Try to kill gracefully first
    kill "$PID" 2>/dev/null

    # Wait a few seconds for graceful shutdown
    for i in {1..10}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "Trading system stopped successfully."
            rm -f "$PID_FILE"
            exit 0
        fi
        sleep 1
    done

    # If still running, force kill
    if kill -0 "$PID" 2>/dev/null; then
        echo "Trading system did not stop gracefully, forcing..."
        kill -9 "$PID" 2>/dev/null
        rm -f "$PID_FILE"
        echo "Trading system force-stopped."
    fi
else
    echo "No PID file found. Trading system may not be running."

    # Try to find any running python main.py processes
    PIDS=$(ps aux | grep "[p]ython.*main.py" | awk '{print $2}')
    if [ -n "$PIDS" ]; then
        echo "Found running processes: $PIDS"
        echo "Stopping them..."
        kill $PIDS 2>/dev/null
        sleep 2
        # Force kill any remaining
        kill -9 $PIDS 2>/dev/null
        echo "Stopped any remaining processes."
    else
        echo "No trading system processes found running."
    fi
fi