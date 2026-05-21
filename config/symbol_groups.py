MAJORS = [
    "BTC/USDT",
    "ETH/USDT",
]

HIGH_LIQUIDITY_ALTS = [
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
]

MIDCAPS = [
    "LINK/USDT",
    "AVAX/USDT",
    "ADA/USDT",
]

MEME = [
    "DOGE/USDT",
    "PEPE/USDT",
    "WIF/USDT",
]

SYMBOL_GROUPS = {
    "MAJORS": MAJORS,
    "HIGH_LIQUIDITY_ALTS": HIGH_LIQUIDITY_ALTS,
    "MIDCAPS": MIDCAPS,
    "MEME": MEME,
}

ALL_GROUPED_SYMBOLS = [
    symbol
    for group in SYMBOL_GROUPS.values()
    for symbol in group
]
