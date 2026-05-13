"""
Phase 5: Observability - Structured Logging & Analytics

Complete audit trail with structured JSON logging for all critical events.
Enable post-trade analysis and performance attribution.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path


class StructuredLogger:
    """
    Structured logging for trading system.
    Every event logged as JSON for analysis.
    """

    def __init__(self, log_dir: str = "logs", name: str = "trading"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # JSON log handler
        json_handler = logging.FileHandler(
            self.log_dir / f"{name}_structured.jsonl"
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(json_handler)

    def log_event(self, event_type: str, **kwargs) -> None:
        """Log structured event"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **kwargs,
        }
        self.logger.info(json.dumps(log_entry))

    def log_trade_entry(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        leverage: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        reason: str,
    ) -> None:
        """Log trade entry with full context"""
        self.log_event(
            "trade_entry",
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=entry_price,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            model_confidence=confidence,
            reason=reason,
        )

    def log_trade_exit(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        realized_pnl: float,
        realized_pnl_pct: float,
        reason: str,
        bars_held: int,
    ) -> None:
        """Log trade exit with PnL"""
        self.log_event(
            "trade_exit",
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            qty=qty,
            realized_pnl_usd=realized_pnl,
            realized_pnl_pct=realized_pnl_pct,
            bars_held=bars_held,
            reason=reason,
        )

    def log_signal_rejected(
        self,
        symbol: str,
        side: str,
        confidence: float,
        reason: str,
        filters_failed: Dict[str, bool],
    ) -> None:
        """Log rejected signal with filters"""
        self.log_event(
            "signal_rejected",
            symbol=symbol,
            proposed_side=side,
            model_confidence=confidence,
            reason=reason,
            filters_failed=filters_failed,
        )

    def log_risk_guard_triggered(
        self,
        guard_type: str,
        severity: str,
        message: str,
        **details,
    ) -> None:
        """Log risk guard activation"""
        self.log_event(
            "risk_guard_triggered",
            guard_type=guard_type,
            severity=severity,
            message=message,
            **details,
        )

    def log_balance_update(
        self,
        previous_balance: float,
        current_balance: float,
        pnl: float,
        pnl_pct: float,
        reason: str,
    ) -> None:
        """Log balance changes"""
        self.log_event(
            "balance_update",
            previous_balance=previous_balance,
            current_balance=current_balance,
            pnl_usd=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
        )


class PerformanceAnalytics:
    """
    Analytics computed from trade history.
    Enables post-trade analysis and performance attribution.
    """

    def __init__(self, trades: list):
        """
        Args:
            trades: List of completed trade records
        """
        self.trades = trades

    def calculate_expectancy(self) -> Dict[str, float]:
        """Calculate mathematical expectancy"""
        if not self.trades:
            return {"expectancy": 0.0, "trades": 0}

        wins = sum(1 for t in self.trades if t.get("realized_pnl", 0) > 0)
        losses = sum(1 for t in self.trades if t.get("realized_pnl", 0) < 0)
        total = len(self.trades)

        win_rate = wins / total if total > 0 else 0
        loss_rate = losses / total if total > 0 else 0

        avg_win = (
            sum(t["realized_pnl"] for t in self.trades if t["realized_pnl"] > 0) / wins
            if wins > 0
            else 0
        )
        avg_loss = (
            sum(t["realized_pnl"] for t in self.trades if t["realized_pnl"] < 0) / losses
            if losses > 0
            else 0
        )

        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

        return {
            "expectancy_per_trade": expectancy,
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_trades": total,
        }

    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from returns"""
        if len(self.trades) < 2:
            return 0.0

        returns = [t.get("realized_pnl_pct", 0) for t in self.trades]

        import statistics

        mean_return = statistics.mean(returns)
        std_dev = statistics.stdev(returns) if len(returns) > 1 else 0.01

        # Annualize (assuming 252 trading days, ~5 trades/day)
        annualization_factor = (252 * 5 / len(self.trades)) ** 0.5

        return (mean_return / std_dev * annualization_factor) if std_dev > 0 else 0.0

    def calculate_max_drawdown(self) -> Dict[str, float]:
        """Calculate maximum drawdown from equity curve"""
        if not self.trades:
            return {"max_drawdown_pct": 0.0}

        equity = 10000  # Starting equity
        equity_curve = [equity]

        for trade in self.trades:
            equity += trade.get("realized_pnl", 0)
            equity_curve.append(equity)

        max_equity = equity_curve[0]
        max_dd = 0.0

        for eq in equity_curve:
            drawdown = (max_equity - eq) / max_equity if max_equity > 0 else 0
            max_dd = max(max_dd, drawdown)
            max_equity = max(max_equity, eq)

        return {"max_drawdown_pct": max_dd}

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get all performance metrics"""
        expectancy = self.calculate_expectancy()
        sharpe = self.calculate_sharpe_ratio()
        max_dd = self.calculate_max_drawdown()

        total_pnl = sum(t.get("realized_pnl", 0) for t in self.trades)
        avg_pnl = total_pnl / len(self.trades) if self.trades else 0

        return {
            "total_trades": len(self.trades),
            "total_pnl_usd": total_pnl,
            "avg_pnl_per_trade": avg_pnl,
            "win_rate": expectancy.get("win_rate", 0),
            "expectancy": expectancy.get("expectancy_per_trade", 0),
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd.get("max_drawdown_pct", 0),
        }
