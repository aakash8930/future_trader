# Exit Logic Refactoring: TP1-Triggered Trailing Model

## Overview
Refactored the exit logic to fix the issue where trailing stops activate too early, causing winning trades to be stopped out on normal pullbacks. The new model uses a fixed TP1 (take-profit) level to trigger profit-lock mode, then manages trailing from a protected profit zone instead of from early partial gains.

## Problem Solved
**Before:** 
- Trailing stop activated after just 1.5 * ATR of profit
- Stop loss would move to breakeven quickly
- Normal pullbacks would trigger stop-outs on profitable trades
- Trailing behaved like an aggressive moving stop-loss

**After:**
- No trailing until TP1 is reached
- Profit-lock mode triggered only when TP1 is hit
- Stop loss protected in a profit zone (TP1 - 0.5 * ATR)
- Trailing only manages further profit extraction after TP1

## Implementation Details

### Files Modified
1. **`execution/runner.py`** - Live trading runner
2. **`backtest/simulator.py`** - Historical backtest simulator

### State Variables Changes

#### Removed
```python
_trail_activated: bool = False
```

#### Added
```python
_profit_lock_mode: bool = False         # Activated when TP1 is hit
profit_lock_sl: float | None = None     # Protected profit zone SL level
```

### Position Management Logic

#### New Exit Behavior
```python
# Check TP1 first (before anything else)
if price >= self.take_profit and not self._profit_lock_mode:
    # TP1 HIT → activate profit-lock mode
    self._profit_lock_mode = True
    # Calculate protected profit zone: TP1 - 0.5 * ATR
    self.profit_lock_sl = self.take_profit - 0.5 * atr
    self.stop_loss = self.profit_lock_sl
    print(f"✅ TP1 HIT → profit lock activated")
    print(f"💰 PROFIT LOCK SL → {price_formatted}")

# Trailing stop (only after TP1 is hit)
if self._profit_lock_mode:
    # After TP1, use trailing stop (only moves upward, never downward)
    new_sl = price - self.cfg.trail_atr_mult * atr
    if new_sl > self.stop_loss:
        self.stop_loss = new_sl
        print(f"📈 TRAILING SL → {price_formatted}")
```

#### Entry Reset
```python
# Reset profit-lock state on new entries
self._profit_lock_mode = False
self.profit_lock_sl = None
```

#### Position Close
```python
# Reset profit-lock state when position closes
self._profit_lock_mode = False
self.profit_lock_sl = None
```

### Pyramiding Behavior
- **Before:** Pyramiding only after initial breakeven protection (trail_activated)
- **After:** Pyramiding only after profit-lock mode activated (more conservative)

## Key Parameters (Unchanged)

Configuration in `strategy.py` remains intact:
```python
trail_atr_mult:          float = 1.35  # trail distance below peak (after TP1)
take_atr_mult:           float = 3.0   # TP1 level = entry + 3.0 * ATR
stop_atr_mult:           float = 2.0   # Initial SL = entry - 2.0 * ATR
```

New calculation (when TP1 hit):
```
profit_lock_sl = take_profit - 0.5 * atr
                = (entry_price + 3.0 * atr) - 0.5 * atr
                = entry_price + 2.5 * atr
```

## Trade Lifecycle Under New Model

1. **Entry**
   - Initial SL = entry - 2.0 * ATR (unchanged)
   - TP1 target = entry + 3.0 * ATR (unchanged)
   - No trailing yet

2. **Move to TP1**
   - Stop loss stays at initial level
   - No early trailing
   - Trade can breathe without getting stopped out

3. **TP1 Hit (Profit Lock Activate)**
   - New SL = (entry + 3.0 * ATR) - 0.5 * ATR = entry + 2.5 * ATR
   - Logging: `✅ TP1 HIT → profit lock activated`
   - Logging: `💰 PROFIT LOCK SL → <value>`

4. **Profit Lock Mode (Trailing)**
   - Trailing adjusts SL = current_price - 1.35 * atr
   - Only moves stop upward (downward moves ignored)
   - Logging: `📈 TRAILING SL → <value>` (when SL moves up)

5. **Exit**
   - Stop hit: Logging shows stop_loss exit
   - Trailing stop manages profit extraction with ATR buffer

## Benefits

✅ Prevents early stop-outs on normal pullbacks
✅ Trades get space to develop profitably
✅ Profit-locked position is protected above break-even
✅ Trailing stop only engages after confirmed profit
✅ ATR-responsive (adjusts for market volatility)
✅ Clear logging for debugging and optimization

## Backward Compatibility

- ✅ Entry logic unchanged
- ✅ Coin selection unchanged
- ✅ Signal filtering unchanged
- ✅ Risk sizing unchanged
- ✅ Threshold adjustments unchanged
- ✅ Only exit management refactored
- ✅ Backtest simulator updated to match runner

## Testing Notes

Both files have been validated for:
- Python syntax correctness
- Proper state initialization
- Logical flow consistency
- Logging output clarity

The refactoring is ready for:
- Backtesting with historical data
- Live/shadow trading
- Performance comparison vs. old model
