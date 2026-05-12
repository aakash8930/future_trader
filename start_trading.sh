#!/bin/bash
# Enhanced script to start the trading system with automatic directory creation,
# proper environment loading, dynamic symbol selection from live market, and self-healing capabilities

# Get the absolute path of the script's directory (the future_trader directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create a run directory with a meaningful name (using timestamp)
RUN_DIR="${SCRIPT_DIR}/trading_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

# Change to the run directory
cd "$RUN_DIR"

# Source the .env file from the original directory properly (handle inline comments and spaces)
if [ -f "${SCRIPT_DIR}/.env" ]; then
    # Read the .env file line by line, export only VAR=VALUE lines
    while IFS='=' read -r key value; do
        # Skip empty lines and comments
        if [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]]; then
            continue
        fi
        # Remove leading/trailing whitespace from key
        key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        # Remove leading/trailing whitespace from value
        value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        # Export the variable
        export "$key=$value"
    done < "${SCRIPT_DIR}/.env"
fi

# Set the PYTHONPATH to the original directory so that the code can be found
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Log file for the script itself (in the run directory)
SCRIPT_LOG="${RUN_DIR}/trading_system.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$DATE] Starting trading system..." >> "$SCRIPT_LOG"
echo "[$DATE] Run directory: $RUN_DIR" >> "$SCRIPT_LOG"

# Check if we need to set up trading symbols from live market
if [ -z "$TRADING_SYMBOLS" ]; then
    echo "[$DATE] TRADING_SYMBOLS is empty, fetching symbols from live market..." >> "$SCRIPT_LOG"

    # Python script to fetch symbols from exchange
    PYTHON_SCRIPT=$(cat << 'EOF'
import os
import sys
from data.fetcher import MarketDataFetcher

try:
    # Create fetcher to get exchange symbols
    fetcher = MarketDataFetcher(
        exchange_name=os.getenv("EXCHANGE_NAME", "binance"),
        fallback_exchanges=os.getenv("EXCHANGE_FALLBACKS", "bybit,kraken,okx").split(","),
        timeout_ms=int(os.getenv("EXCHANGE_TIMEOUT_MS", "20000"))
    )

    # Get all USDT pairs (assuming we want to trade against USDT)
    all_symbols = list(fetcher.supported_symbols)
    usdt_symbols = [s for s in all_symbols if s.endswith('/USDT')]

    # If we don't have enough USDT pairs, fall back to some major ones
    if len(usdt_symbols) < 5:
        usdt_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT", "DOGE/USDT", "BNB/USDT"]

    # Limit to a reasonable number to avoid too many symbols (top 30 by alphabetical order for consistency)
    # In a production system, we might want to sort by volume or market cap instead
    usdt_symbols.sort()
    selected_symbols = usdt_symbols[:30]

    # Output the symbols as comma-separated string
    print(",".join(selected_symbols))

except Exception as e:
    print(f"Error fetching symbols from exchange: {e}", file=sys.stderr)
    # Fallback to default symbols
    print("BTC/USDT,ETH/USDT,SOL/USDT,AVAX/USDT,LINK/USDT,DOGE/USDT,BNB/USDT")
EOF
)

    # Execute the Python script to get symbols
    # Redirect stderr to /dev/null to suppress error messages (we'll get fallback output on stdout)
    # Pipe stdout through grep to remove lines that start with "[" (fetcher debug messages)
    FETCHED_SYMBOLS=$(python3 -c "$PYTHON_SCRIPT" 2>/dev/null | grep -v "^\\[")

    # Check if we got a valid response (should contain at least one comma or be a single symbol)
    if [ -z "$FETCHED_SYMBOLS" ] || [[ ! "$FETCHED_SYMBOLS" =~ [A-Z0-9]+/[A-Z0-9]+ ]]; then
        # Fallback to default symbols if the fetch failed
        FETCHED_SYMBOLS="BTC/USDT,ETH/USDT,SOL/USDT,AVAX/USDT,LINK/USDT,DOGE/USDT,BNB/USDT"
        echo "[$DATE] Failed to fetch symbols from exchange, using default symbols" >> "$SCRIPT_LOG"
    fi

    # Set the TRADING_SYMBOLS environment variable
    export TRADING_SYMBOLS="$FETCHED_SYMBOLS"

    echo "[$DATE] Fetched symbols from live market: $TRADING_SYMBOLS" >> "$SCRIPT_LOG"
else
    echo "[$DATE] Using configured symbols: $TRADING_SYMBOLS" >> "$SCRIPT_LOG"
fi

# Check if we need to train models (if missing)
echo "[$DATE] Checking for required models..." >> "$SCRIPT_LOG"

# Function to check if model exists for a symbol
check_model_exists() {
    local symbol=$1
    local symbol_dir="${SCRIPT_DIR}/models/${symbol}"
    if [ -d "$symbol_dir" ] && [ -f "$symbol_dir/model.pt" ] && [ -f "$symbol_dir/scaler.save" ] && [ -f "$symbol_dir/metadata.json" ]; then
        return 0
    else
        return 1
    fi
}

# Function to train models for missing symbols
train_missing_models() {
    local missing_symbols=()
    # Get symbols from environment
    local symbols=${TRADING_SYMBOLS:-""}
    if [ -z "$symbols" ]; then
        echo "[$DATE] No symbols to check for models" >> "$SCRIPT_LOG"
        return
    fi
    IFS=',' read -ra SYMBOL_ARRAY <<< "$symbols"
    for symbol in "${SYMBOL_ARRAY[@]}"; do
        # Clean up symbol (remove spaces, convert to filename format)
        clean_symbol=$(echo "$symbol" | tr '/' '_' | tr -d '[:space:]')
        if ! check_model_exists "$clean_symbol"; then
            missing_symbols+=("$symbol")
        fi
    done

    if [ ${#missing_symbols[@]} -gt 0 ]; then
        echo "[$DATE] Missing models for: ${missing_symbols[*]}" >> "$SCRIPT_LOG"
        echo "[$DATE] Attempting to train missing models..." >> "$SCRIPT_LOG"

        # Train each missing symbol
        for symbol in "${missing_symbols[@]}"; do
            clean_symbol=$(echo "$symbol" | tr '/' '_' | tr -d '[:space:]')
            echo "[$DATE] Training model for $symbol..." >> "$SCRIPT_LOG"
            # Run training in background to not block startup, but we'll wait for completion
            TRAIN_SYMBOLS="$symbol" python -m train.train_direction_model >> "$SCRIPT_LOG" 2>&1
            if [ $? -eq 0 ]; then
                echo "[$DATE] Successfully trained model for $symbol" >> "$SCRIPT_LOG"
            else
                echo "[$DATE] Failed to train model for $symbol" >> "$SCRIPT_LOG"
            fi
        done
    else
        echo "[$DATE] All required models are present" >> "$SCRIPT_LOG"
    fi
}

# Check and train models if needed
train_missing_models

# Start the trading system with output logging
echo "[$DATE] Launching trading engine..." >> "$SCRIPT_LOG"
nohup python -u "${SCRIPT_DIR}/main.py" >> "$SCRIPT_LOG" 2>&1 &

# Get the process ID
PID=$!

# Save the PID to a file in the run directory
echo $PID > "${RUN_DIR}/trading_system.pid"

echo "[$DATE] Started trading system with PID $PID" >> "$SCRIPT_LOG"
echo "Trading system started. PID: $PID"
echo "Log file: $SCRIPT_LOG"
echo "Run directory: $RUN_DIR"
echo "To stop: ${RUN_DIR}/stop_trading.sh"

# Create a stop script in the run directory
STOP_SCRIPT="${RUN_DIR}/stop_trading.sh"
cat > "$STOP_SCRIPT" << 'EOF'
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
EOF

# Make the stop script executable
chmod +x "$STOP_SCRIPT"