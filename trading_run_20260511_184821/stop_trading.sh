#!/bin/bash
# Stop script for the trading system run in this directory

# Get the current directory (the run directory)
RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PID_FILE="${RUN_DIR}/trading_system.pid"

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

    # Try to find any python main.py processes that are associated with this run
    # Look for processes with this run directory in their command line or working directory
    if [ -n "$RUN_DIR" ]; then
        PIDS=$(ps aux | grep "[p]ython.*main.py" | grep "$RUN_DIR" | awk '{print $2}')
        if [ -n "$PIDS" ]; then
            echo "Found running processes associated with this run: $PIDS"
            echo "Stopping them..."
            kill $PIDS 2>/dev/null
            sleep 2
            kill -9 $PIDS 2>/dev/null
            echo "Stopped any remaining processes."
        else
            echo "No specific processes found for this run."
        fi
    else
        echo "RUN_DIR not set, checking for any python main.py processes..."
        PIDS=$(ps aux | grep "[p]ython.*main.py" | awk '{print $2}')
        if [ -n "$PIDS" ]; then
            echo "Found running processes: $PIDS"
            echo "Stopping them..."
            kill $PIDS 2>/dev/null
            sleep 2
            kill -9 $PIDS 2>/dev/null
            echo "Stopped any remaining processes."
        else
            echo "No trading system processes found running."
        fi
    fi
fi
