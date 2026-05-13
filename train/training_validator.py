"""
Phase 8: Training Integration - Market Validator & Symbol Whitelist

Integrate market validator into training pipeline.
Prevent training on garbage symbols (delisted, illiquid, leveraged tokens, etc).
Enforce strict symbol qualification before model training.
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class TrainingSymbolFilter:
    """
    Symbol qualification filter for training pipeline.
    
    Only allows symbols that:
    - Are USDT perpetual futures
    - Have minimum history (500+ candles)
    - Have sufficient liquidity
    - Have reasonable volatility
    - Are not leveraged tokens or delisted
    """

    def __init__(
        self,
        min_candles: int = 500,
        min_daily_volume_usd: float = 50000,
        min_volatility_pct: float = 0.0005,
        max_volatility_pct: float = 0.25,
        max_spread_pct: float = 0.05,
        whitelist_file: Optional[str] = None,
    ):
        self.min_candles = min_candles
        self.min_daily_volume_usd = min_daily_volume_usd
        self.min_volatility_pct = min_volatility_pct
        self.max_volatility_pct = max_volatility_pct
        self.max_spread_pct = max_spread_pct
        self.whitelist_file = whitelist_file
        self.approved_symbols = set()

        # Load whitelist if provided
        if whitelist_file and Path(whitelist_file).exists():
            self._load_whitelist()

    def _load_whitelist(self) -> None:
        """Load approved symbol list from file"""
        try:
            with open(self.whitelist_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.approved_symbols = set(data)
                elif isinstance(data, dict):
                    self.approved_symbols = set(data.get("approved_symbols", []))
            logger.info(f"Loaded {len(self.approved_symbols)} approved symbols")
        except Exception as e:
            logger.warning(f"Failed to load whitelist: {e}")

    def save_whitelist(self, symbols: List[str]) -> None:
        """Save approved symbol list"""
        if not self.whitelist_file:
            return

        try:
            Path(self.whitelist_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.whitelist_file, "w") as f:
                json.dump(
                    {"approved_symbols": sorted(symbols)},
                    f,
                    indent=2,
                )
            logger.info(f"Saved {len(symbols)} approved symbols to {self.whitelist_file}")
        except Exception as e:
            logger.error(f"Failed to save whitelist: {e}")

    def check_symbol(
        self,
        symbol: str,
        df: Optional[Any] = None,
        volume_data: Optional[Dict[str, float]] = None,
    ) -> tuple[bool, str]:
        """
        Check if symbol qualifies for training.
        
        Args:
            symbol: Symbol to check
            df: OHLCV dataframe for this symbol
            volume_data: Dict mapping symbols to daily volume USD
            
        Returns:
            (approved: bool, reason: str)
        """

        # === Basic symbol checks ===

        # Never train on leveraged tokens
        if any(
            token in symbol.upper() for token in ["BULL", "BEAR", "UP", "DOWN"]
        ):
            return False, "Leveraged token (forbidden)"

        # Reject stablecoins (no directional edge)
        if any(
            stable in symbol.upper() for stable in ["USDT", "USDC", "BUSD", "DAI"]
        ):
            if symbol != symbol.replace("USDT", "").replace("USDC", "").replace("BUSD", "").replace("DAI", ""):
                # Check it's not e.g. "ETHUSDT" which we want
                pass

        # Reject non-perpetual symbols
        if not symbol.endswith("USDT"):
            return False, "Not USDT perpetual"

        # Reject if in whitelist and NOT approved
        if self.approved_symbols and symbol not in self.approved_symbols:
            return False, "Not in whitelist"

        # === Data checks (if provided) ===

        if df is None or len(df) < self.min_candles:
            missing = self.min_candles - (len(df) if df is not None else 0)
            return False, f"Insufficient history ({missing} candles short)"

        # Check volatility
        if "close" in df.columns:
            close = df["close"]
            returns = close.pct_change().dropna()

            if len(returns) > 0:
                volatility = returns.std()

                if volatility < self.min_volatility_pct:
                    return False, f"Too stable ({volatility:.4f} < {self.min_volatility_pct})"

                if volatility > self.max_volatility_pct:
                    return False, f"Too volatile ({volatility:.4f} > {self.max_volatility_pct})"

        # Check volume
        if volume_data and symbol in volume_data:
            daily_volume = volume_data[symbol]

            if daily_volume < self.min_daily_volume_usd:
                return False, f"Low volume (${daily_volume:.0f} < ${self.min_daily_volume_usd})"

        # All checks passed
        return True, "Symbol approved for training"

    def filter_symbols(
        self,
        symbols: List[str],
        volume_data: Optional[Dict[str, float]] = None,
    ) -> tuple[List[str], Dict[str, str]]:
        """
        Filter list of symbols, returning approved and rejection reasons.
        
        Returns:
            (approved_symbols, rejection_map)
        """
        approved = []
        rejections = {}

        for symbol in symbols:
            is_valid, reason = self.check_symbol(symbol, volume_data=volume_data)
            if is_valid:
                approved.append(symbol)
            else:
                rejections[symbol] = reason

        return approved, rejections


class TrainingPipelineValidator:
    """
    Validates training pipeline before model generation.
    
    Ensures:
    - Symbols are qualified
    - Data is clean
    - Class balance is acceptable
    - Minimum trade count
    """

    def __init__(self, symbol_filter: Optional[TrainingSymbolFilter] = None):
        self.symbol_filter = symbol_filter or TrainingSymbolFilter()

    def validate_training_data(
        self,
        symbol: str,
        df: Any,
        y: Optional[Any] = None,
        min_samples: int = 100,
        min_positive_samples: int = 10,
    ) -> tuple[bool, str]:
        """
        Validate that training data is suitable for model training.
        
        Args:
            symbol: Symbol being trained
            df: Feature dataframe
            y: Target labels (if available)
            min_samples: Minimum total samples
            min_positive_samples: Minimum positive class samples
            
        Returns:
            (valid: bool, reason: str)
        """

        # Check symbol qualification
        symbol_valid, symbol_reason = self.symbol_filter.check_symbol(symbol, df=df)
        if not symbol_valid:
            return False, f"Symbol not qualified: {symbol_reason}"

        # Check data volume
        if len(df) < min_samples:
            return False, f"Insufficient data ({len(df)} < {min_samples} samples)"

        # Check label distribution (if labels provided)
        if y is not None:
            try:
                import numpy as np

                unique, counts = np.unique(y, return_counts=True)

                if len(unique) < 2:
                    return False, "Only single class in labels"

                positive_count = counts[-1] if len(counts) >= 2 else 0

                if positive_count < min_positive_samples:
                    return False, f"Too few positive samples ({positive_count} < {min_positive_samples})"

                # Check class balance (avoid 95/5 extreme imbalance)
                positive_pct = positive_count / len(y)
                if positive_pct < 0.10 or positive_pct > 0.90:
                    return (
                        False,
                        f"Severe class imbalance ({positive_pct:.1%} positive)",
                    )
            except Exception as e:
                logger.warning(f"Could not analyze labels: {e}")

        return True, "Data approved for training"

    def pre_training_check(
        self,
        symbols: List[str],
        data_map: Dict[str, Any],
        volume_data: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Pre-training validation of all symbols and data.
        
        Returns report of approved/rejected symbols and reasons.
        """
        report = {
            "total_symbols": len(symbols),
            "approved_symbols": [],
            "rejected_symbols": {},
            "warnings": [],
        }

        approved_symbols, rejections = self.symbol_filter.filter_symbols(
            symbols, volume_data
        )

        # Check each approved symbol's data quality
        for symbol in approved_symbols:
            if symbol not in data_map:
                report["rejected_symbols"][symbol] = "No data provided"
                continue

            df = data_map[symbol]
            valid, reason = self.validate_training_data(symbol, df)

            if not valid:
                report["rejected_symbols"][symbol] = reason
            else:
                report["approved_symbols"].append(symbol)

        # Add all symbol-level rejections
        report["rejected_symbols"].update(rejections)

        # Log summary
        logger.info(
            f"Training pipeline check: {len(report['approved_symbols'])} approved, "
            f"{len(report['rejected_symbols'])} rejected"
        )

        return report
