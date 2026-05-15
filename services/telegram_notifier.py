"""
Telegram notification service for RUDRA-ALPHA trading system.
Provides async-safe, non-blocking notifications for trade events, errors, and system lifecycle.
"""

import asyncio
import aiohttp
import os
import json
from typing import Optional
from datetime import datetime
from threading import Thread
from queue import Queue, Empty
import logging

# Configuration from environment
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Notification queue (thread-safe)
_notification_queue: Queue = Queue(maxsize=1000)
_worker_thread: Optional[Thread] = None
_stop_event: asyncio.Event = None

# Logging
logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send Telegram messages asynchronously without blocking execution."""
    
    def __init__(self, bot_token: str = "", chat_id: str = "", enabled: bool = True):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Target chat ID
            enabled: Whether to actually send messages
        """
        self.bot_token = bot_token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self.enabled = enabled and bool(self.bot_token and self.chat_id)
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else ""
        self.session: Optional[aiohttp.ClientSession] = None
        
        if not self.enabled:
            logger.warning("[TELEGRAM] Notifications disabled (missing token or chat_id)")
    
    async def _ensure_session(self) -> Optional[aiohttp.ClientSession]:
        """Create or return existing aiohttp session."""
        if self.session is None and self.enabled:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        timeout_sec: float = 10.0,
        retry_count: int = 3
    ) -> bool:
        """
        Send a Telegram message with retry logic.
        
        Args:
            text: Message text
            parse_mode: HTML or Markdown
            timeout_sec: Request timeout in seconds
            retry_count: Number of retries on failure
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        if not text:
            logger.warning("[TELEGRAM] Empty message text")
            return False
        
        session = await self._ensure_session()
        if not session:
            logger.error("[TELEGRAM] Failed to create session")
            return False
        
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        
        for attempt in range(retry_count):
            try:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_sec)
                ) as resp:
                    if resp.status == 200:
                        logger.debug(f"[TELEGRAM] Message sent successfully")
                        return True
                    else:
                        error_text = await resp.text()
                        logger.warning(f"[TELEGRAM] HTTP {resp.status}: {error_text}")
                        
                        if attempt < retry_count - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except asyncio.TimeoutError:
                logger.warning(f"[TELEGRAM] Timeout on attempt {attempt + 1}/{retry_count}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"[TELEGRAM] Error on attempt {attempt + 1}/{retry_count}: {e}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
        
        logger.error(f"[TELEGRAM] Failed to send message after {retry_count} retries")
        return False
    
    async def close(self):
        """Close the session."""
        if self.session:
            await self.session.close()
            self.session = None


# Global notifier instance
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Get or create the global Telegram notifier."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier(enabled=TELEGRAM_ENABLED)
    return _notifier


async def _worker_loop():
    """Background worker that processes notification queue."""
    notifier = get_notifier()
    
    while True:
        try:
            # Non-blocking get with timeout
            try:
                notification = _notification_queue.get(timeout=1.0)
            except Empty:
                await asyncio.sleep(0.1)
                continue
            
            if notification is None:  # Poison pill to stop
                break
            
            msg_type, text, kwargs = notification
            await notifier.send_message(text, **kwargs)
            
        except Exception as e:
            logger.error(f"[TELEGRAM WORKER] Error: {e}")
            await asyncio.sleep(1)
    
    await notifier.close()


def _start_worker_thread():
    """Start the background worker thread (non-blocking)."""
    global _worker_thread, _stop_event
    
    if _worker_thread and _worker_thread.is_alive():
        return  # Already running
    
    def run_event_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_worker_loop())
        finally:
            loop.close()
    
    _worker_thread = Thread(target=run_event_loop, daemon=True)
    _worker_thread.start()
    logger.info("[TELEGRAM] Worker thread started")


def _enqueue_notification(msg_type: str, text: str, **kwargs):
    """Add notification to queue for async processing."""
    if not TELEGRAM_ENABLED or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    
    _start_worker_thread()
    
    try:
        _notification_queue.put_nowait((msg_type, text, kwargs))
    except Exception as e:
        logger.warning(f"[TELEGRAM] Failed to queue notification: {e}")


# ─── Public API ────────────────────────────────────────────────────────────


def notify_trade_open(
    symbol: str,
    side: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    qty: float,
    probability: float,
    adx: float,
    mode: str = "paper"
):
    """Notify that a trade was opened."""
    side_emoji = "🟢" if side.upper() == "LONG" else "🔴"
    
    text = f"""{side_emoji} OPEN {side.upper()} {symbol}

<b>Entry:</b> {entry_price:.2f}
<b>SL:</b> {stop_loss:.2f}
<b>TP:</b> {take_profit:.2f}
<b>Qty:</b> {qty:.6f}
<b>Prob:</b> {probability:.2%}
<b>ADX:</b> {adx:.1f}
<b>Mode:</b> {mode}"""
    
    _enqueue_notification("trade_open", text)


def notify_trade_close(
    symbol: str,
    side: str,
    pnl: float,
    pnl_pct: float,
    balance: float,
    duration_minutes: int,
    close_reason: str,
    mode: str = "paper"
):
    """Notify that a trade was closed."""
    side_emoji = "🟢" if side.upper() == "LONG" else "🔴"
    pnl_emoji = "✅" if pnl > 0 else "❌" if pnl < 0 else "⚪"
    
    text = f"""{pnl_emoji} CLOSE {side.upper()} {symbol}

<b>Reason:</b> {close_reason}
<b>PnL:</b> {pnl:+.4f} ({pnl_pct:+.2%})
<b>Balance:</b> {balance:.2f}
<b>Duration:</b> {duration_minutes}m
<b>Mode:</b> {mode}"""
    
    _enqueue_notification("trade_close", text)


def notify_take_profit(
    symbol: str,
    side: str,
    pnl: float,
    mode: str = "paper"
):
    """Notify that take profit was hit."""
    text = f"""🎯 TAKE PROFIT {side.upper()} {symbol}

<b>PnL:</b> {pnl:+.4f}
<b>Mode:</b> {mode}"""
    
    _enqueue_notification("take_profit", text)


def notify_stop_loss(
    symbol: str,
    side: str,
    pnl: float,
    mode: str = "paper"
):
    """Notify that stop loss was hit."""
    text = f"""🛑 STOP LOSS {side.upper()} {symbol}

<b>PnL:</b> {pnl:+.4f}
<b>Mode:</b> {mode}"""
    
    _enqueue_notification("stop_loss", text)


def notify_trailing_stop(
    symbol: str,
    side: str,
    pnl: float,
    mode: str = "paper"
):
    """Notify that trailing stop was executed."""
    text = f"""📉 TRAILING STOP {side.upper()} {symbol}

<b>PnL:</b> {pnl:+.4f}
<b>Mode:</b> {mode}"""
    
    _enqueue_notification("trailing_stop", text)


def notify_market_guard_activated(
    symbol: str,
    reason: str,
    price: float,
    time_str: str = ""
):
    """Notify that market guard was activated."""
    text = f"""⚠️ MARKET GUARD ACTIVATED {symbol}

<b>Reason:</b> {reason}
<b>Price:</b> {price:.2f}
<b>Time:</b> {time_str or datetime.utcnow().isoformat()}"""
    
    _enqueue_notification("market_guard", text)


def notify_error(
    error_type: str,
    message: str,
    traceback_str: str = ""
):
    """Notify about a runtime or database error."""
    text = f"""❌ {error_type.upper()}

<b>Message:</b> <code>{message}</code>"""
    
    if traceback_str:
        text += f"\n\n<b>Traceback:</b>\n<code>{traceback_str[:500]}</code>"
    
    _enqueue_notification("error", text)


def notify_startup(symbol: str, mode: str = "paper", timeframe: str = "1h"):
    """Notify that system started."""
    text = f"""🚀 SYSTEM STARTUP

<b>Symbol:</b> {symbol}
<b>Mode:</b> {mode}
<b>Timeframe:</b> {timeframe}
<b>Time:</b> {datetime.utcnow().isoformat()}Z"""
    
    _enqueue_notification("startup", text)


def notify_shutdown(symbol: str, reason: str = ""):
    """Notify that system shut down."""
    text = f"""🛑 SYSTEM SHUTDOWN

<b>Symbol:</b> {symbol}
<b>Reason:</b> {reason or "User requested"}
<b>Time:</b> {datetime.utcnow().isoformat()}Z"""
    
    _enqueue_notification("shutdown", text)


def notify_reconnect(symbol: str, attempt: int = 1):
    """Notify about reconnection attempt."""
    text = f"""🔗 RECONNECT ATTEMPT {symbol}

<b>Attempt:</b> {attempt}
<b>Time:</b> {datetime.utcnow().isoformat()}Z"""
    
    _enqueue_notification("reconnect", text)


def notify_position_reversal(
    symbol: str,
    from_side: str,
    to_side: str,
    mode: str = "paper"
):
    """Notify about position reversal."""
    text = f"""🔄 POSITION REVERSAL {symbol}

<b>From:</b> {from_side.upper()}
<b>To:</b> {to_side.upper()}
<b>Mode:</b> {mode}
<b>Time:</b> {datetime.utcnow().isoformat()}Z"""
    
    _enqueue_notification("reversal", text)


def notify_manual_pause(symbol: str):
    """Notify about manual pause."""
    text = f"""⏸️ TRADING PAUSED {symbol}

<b>Time:</b> {datetime.utcnow().isoformat()}Z"""
    
    _enqueue_notification("pause", text)


def notify_manual_resume(symbol: str):
    """Notify about manual resume."""
    text = f"""▶️ TRADING RESUMED {symbol}

<b>Time:</b> {datetime.utcnow().isoformat()}Z"""
    
    _enqueue_notification("resume", text)


def shutdown():
    """Gracefully shut down the notification system."""
    global _worker_thread
    
    if _worker_thread and _worker_thread.is_alive():
        _notification_queue.put(None)  # Poison pill
        _worker_thread.join(timeout=5)
        logger.info("[TELEGRAM] Worker thread shut down")
