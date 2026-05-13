# data/stale_detector.py
"""
Stale Data Detector - Monitor price data freshness and detect stalled streams.

Prevents trading with stale prices which can lead to slippage, bad fills, 
and incorrect PnL calculations.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DataFreshness:
    """Data freshness state for a symbol."""
    symbol: str
    last_update: datetime
    max_age_seconds: int
    is_fresh: bool
    age_seconds: float
    warning_message: Optional[str] = None


class StaleDataDetector:
    """
    Monitor price data freshness and detect stalled WebSocket streams.
    
    Tracks when last data was received for each symbol and rejects trades
    if data is too old (stale).
    """
    
    def __init__(
        self,
        max_age_seconds: int = 60,
        warn_threshold_seconds: int = 30,
    ):
        """
        Initialize stale data detector.
        
        Args:
            max_age_seconds: Max allowed age before data considered stale (default 60)
            warn_threshold_seconds: Warn if data older than this (default 30)
        """
        self.max_age_seconds = max_age_seconds
        self.warn_threshold_seconds = warn_threshold_seconds
        
        # Track last update time per symbol
        self.last_update: Dict[str, datetime] = {}
        
        # Track stale events
        self.stale_events: List[str] = []
        self.warning_events: List[str] = []
    
    def update_timestamp(self, symbol: str, timestamp: Optional[datetime] = None):
        """
        Record data update for a symbol.
        
        Args:
            symbol: Trading symbol
            timestamp: Data timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.last_update[symbol] = timestamp
    
    def is_data_fresh(self, symbol: str) -> Tuple[bool, Optional[str]]:
        """
        Check if data for a symbol is fresh.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            (is_fresh, reason) tuple
            - is_fresh: True if data is recent
            - reason: Reason if stale (None if fresh)
        """
        if symbol not in self.last_update:
            reason = f"{symbol}: No data received yet"
            self.stale_events.append(reason)
            logger.warning(f"[STALE DETECTOR] {reason}")
            return (False, reason)
        
        last_time = self.last_update[symbol]
        now = datetime.utcnow()
        age = (now - last_time).total_seconds()
        
        # Check if stale
        if age > self.max_age_seconds:
            reason = (
                f"{symbol}: Data is {age:.1f}s old (max: {self.max_age_seconds}s). "
                "Refusing to trade with stale data."
            )
            self.stale_events.append(reason)
            logger.error(f"[STALE DETECTOR] {reason}")
            return (False, reason)
        
        # Warn if approaching stale
        if age > self.warn_threshold_seconds:
            warn_msg = (
                f"{symbol}: Data is {age:.1f}s old (warning threshold: {self.warn_threshold_seconds}s). "
                "Monitor data source."
            )
            self.warning_events.append(warn_msg)
            logger.warning(f"[STALE DETECTOR] {warn_msg}")
        
        return (True, None)
    
    def get_freshness_report(self, symbols: List[str]) -> List[DataFreshness]:
        """
        Get freshness status for all symbols.
        
        Args:
            symbols: List of symbols to check
            
        Returns:
            List of DataFreshness objects with status
        """
        now = datetime.utcnow()
        report = []
        
        for symbol in symbols:
            if symbol not in self.last_update:
                freshness = DataFreshness(
                    symbol=symbol,
                    last_update=None,
                    max_age_seconds=self.max_age_seconds,
                    is_fresh=False,
                    age_seconds=float('inf'),
                    warning_message="No data received",
                )
            else:
                last_time = self.last_update[symbol]
                age = (now - last_time).total_seconds()
                is_fresh = age <= self.max_age_seconds
                warning = None
                
                if age > self.max_age_seconds:
                    warning = f"Data is {age:.1f}s stale"
                elif age > self.warn_threshold_seconds:
                    warning = f"Data is {age:.1f}s old (approaching stale)"
                
                freshness = DataFreshness(
                    symbol=symbol,
                    last_update=last_time,
                    max_age_seconds=self.max_age_seconds,
                    is_fresh=is_fresh,
                    age_seconds=age,
                    warning_message=warning,
                )
            
            report.append(freshness)
        
        return report
    
    def get_stale_symbols(self, symbols: List[str]) -> List[str]:
        """
        Get list of symbols with stale data.
        
        Args:
            symbols: List of symbols to check
            
        Returns:
            List of symbols with stale data
        """
        stale = []
        for symbol in symbols:
            is_fresh, _ = self.is_data_fresh(symbol)
            if not is_fresh:
                stale.append(symbol)
        
        return stale
    
    def assert_data_fresh(self, symbol: str) -> None:
        """
        Assert that data is fresh, raise exception if stale.
        
        Args:
            symbol: Trading symbol
            
        Raises:
            ValueError: If data is stale
        """
        is_fresh, reason = self.is_data_fresh(symbol)
        if not is_fresh:
            raise ValueError(f"Cannot trade with stale data: {reason}")
    
    def detect_stream_death(self, symbols: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Detect if WebSocket stream appears dead (all symbols stale).
        
        Args:
            symbols: List of symbols being monitored
            
        Returns:
            (is_stream_alive, reason) tuple
            - is_stream_alive: False if all symbols stale
            - reason: Description of stream death
        """
        if not symbols:
            return (True, None)
        
        stale_symbols = self.get_stale_symbols(symbols)
        
        if len(stale_symbols) == len(symbols):
            reason = (
                f"Stream appears DEAD: All {len(symbols)} symbols have stale data. "
                "WebSocket likely disconnected."
            )
            self.stale_events.append(reason)
            logger.critical(f"[STALE DETECTOR] {reason}")
            return (False, reason)
        
        if len(stale_symbols) > len(symbols) * 0.5:
            warn_msg = (
                f"Stream degraded: {len(stale_symbols)}/{len(symbols)} symbols stale. "
                "Check WebSocket connection."
            )
            self.warning_events.append(warn_msg)
            logger.warning(f"[STALE DETECTOR] {warn_msg}")
        
        return (True, None)
    
    def clear_history(self):
        """Clear event history."""
        self.stale_events = []
        self.warning_events = []
    
    def get_report(self) -> Dict:
        """Get detector report."""
        return {
            "total_stale_events": len(self.stale_events),
            "total_warnings": len(self.warning_events),
            "stale_events": self.stale_events,
            "warning_events": self.warning_events,
            "config": {
                "max_age_seconds": self.max_age_seconds,
                "warn_threshold_seconds": self.warn_threshold_seconds,
            }
        }


def check_data_age(
    symbol: str,
    last_update: datetime,
    max_age_seconds: int = 60,
) -> Tuple[bool, float]:
    """
    Quick check of data age.
    
    Args:
        symbol: Trading symbol
        last_update: Timestamp of last data update
        max_age_seconds: Maximum acceptable age
        
    Returns:
        (is_fresh, age_seconds) tuple
    """
    now = datetime.utcnow()
    age = (now - last_update).total_seconds()
    is_fresh = age <= max_age_seconds
    return (is_fresh, age)
