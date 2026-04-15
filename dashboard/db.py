# This module is deprecated — SQL dependencies have been removed.
# All data is now served from JSONL files via logs/trade_store.py
# DO NOT use any function from this module — they all raise NotImplementedError

def _raise():
    raise NotImplementedError("dashboard.db is deprecated — use logs.trade_store instead")


def get_performance_summary(): _raise()
def get_trades(*a, **k): _raise()
def get_equity_curve(*a, **k): _raise()
def get_distinct_symbols(): _raise()
def get_latest_balance(): _raise()