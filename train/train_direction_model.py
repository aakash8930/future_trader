#train/train_direction_model.py

import os
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _load_project_modules():
    from data.fetcher import MarketDataFetcher
    from features.technicals import compute_core_features
    from models.model_identity import MODEL_NAME, MODEL_VERSION

    return MarketDataFetcher, compute_core_features, MODEL_NAME, MODEL_VERSION


# =========================
# CONFIG
# =========================
TIMEFRAME = "15m"
CANDLES = 50000

HORIZON = 12
STOP_ATR_MULT = 2.0
TAKE_ATR_MULT = 3.0

EPOCHS = 10
BATCH_SIZE = 256
LR = 1e-3
TRAIN_SPLIT = 0.8
EARLY_STOPPING_PATIENCE = 3
PURGE_BARS = 10           # leakage protection

FEATURE_COLUMNS = [
    "ema_fast",
    "ema_slow",
    "ema_spread",
    "dist_ema200",
    "ema_fast_slope",
    "rsi",
    "rsi_delta",
    "ret",
    "vol",
    "volume_zscore",
    "atr_pct",
    "adx",
    "breakout_strength",
]

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "MATIC/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "ADA/USDT",
    "XRP/USDT",
    "DOGE/USDT",
]

# allow temporary exclusion of symbols via environment (same variable as live)
raw_exclude = os.getenv("EXCLUDED_SYMBOLS", "")
if raw_exclude:
    excluded = {s.strip().upper() for s in raw_exclude.split(",") if s.strip()}
    SYMBOLS = [s for s in SYMBOLS if s not in excluded]


# =========================
# MODEL
# =========================
class DirectionNet(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def _build_triple_barrier_labels(
    df,
    horizon: int,
    stop_atr_mult: float,
    take_atr_mult: float,
):
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr = df["atr"].to_numpy()

    labels = np.zeros(len(df), dtype=np.int8)
    valid = np.zeros(len(df), dtype=bool)

    for i in range(0, len(df) - horizon):
        entry = close[i]
        atr_i = atr[i]
        if not np.isfinite(entry) or not np.isfinite(atr_i) or atr_i <= 0:
            continue

        tp_level = entry + take_atr_mult * atr_i
        sl_level = entry - stop_atr_mult * atr_i

        future_high = high[i + 1 : i + horizon + 1]
        future_low = low[i + 1 : i + horizon + 1]

        label = 0
        for high_j, low_j in zip(future_high, future_low):
            tp_hit = high_j >= tp_level
            sl_hit = low_j <= sl_level

            # If both hit within a single bar, assume worst-case ordering.
            if tp_hit and sl_hit:
                label = 0
                break
            if tp_hit:
                label = 1
                break
            if sl_hit:
                label = 0
                break

        labels[i] = label
        valid[i] = True

    return labels, valid


def _optimize_long_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict]:
    best_th = 0.55
    best_score = -1e9
    best_stats = {}

    min_pred_trades = max(25, int(0.02 * len(y_true)))

    for th in np.arange(0.45, 0.71, 0.01):
        preds = (y_prob >= th).astype(int)
        pred_count = int(preds.sum())
        if pred_count < min_pred_trades:
            continue

        precision = float(precision_score(y_true, preds, zero_division=0))
        recall = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        trade_rate = float(preds.mean())

        # Precision-biased score with mild turnover penalty.
        score = 0.70 * precision + 0.25 * f1 + 0.05 * recall - max(0.0, trade_rate - 0.35) * 0.10

        if score > best_score:
            best_score = score
            best_th = float(th)
            best_stats = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "trade_rate": trade_rate,
                "score": float(score),
            }

    return best_th, best_stats


# =========================
# TRAINING
# =========================
def train_for_symbol(symbol: str):
    MarketDataFetcher, compute_core_features, MODEL_NAME, MODEL_VERSION = _load_project_modules()
    print(f"\n🚀 Training {MODEL_NAME} {MODEL_VERSION} for {symbol}")

    exchange_name = os.getenv("EXCHANGE_NAME", "binance")
    raw_fallbacks = os.getenv("EXCHANGE_FALLBACKS", "bybit,kraken,okx")
    fallbacks = [s.strip().lower() for s in raw_fallbacks.split(",") if s.strip()]
    timeout_ms = int(os.getenv("EXCHANGE_TIMEOUT_MS", "20000"))

    fetcher = MarketDataFetcher(
        exchange_name=exchange_name,
        fallback_exchanges=fallbacks,
        timeout_ms=timeout_ms,
    )
    df = fetcher.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLES)

    df = compute_core_features(df)

    # -------------------------
    # Triple-barrier target:
    # 1 if TP is hit before SL within horizon, else 0.
    # -------------------------
    labels, valid = _build_triple_barrier_labels(
        df,
        horizon=HORIZON,
        stop_atr_mult=STOP_ATR_MULT,
        take_atr_mult=TAKE_ATR_MULT,
    )
    df["target"] = labels
    df = df[valid].copy()

    df.dropna(inplace=True)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df["target"].values.astype(np.float32).reshape(-1, 1)

    # -------------------------
    # Train / Validation split with purge gap
    # -------------------------
    split_idx = int(len(df) * TRAIN_SPLIT)
    train_end = max(0, split_idx - PURGE_BARS)

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[split_idx:]
    y_val = y[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # -------------------------
    # Handle class imbalance
    # -------------------------
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    y_train_flat = y_train.flatten()
    class_counts = np.bincount(y_train_flat.astype(int), minlength=2)

    if class_counts.min() == 0:
        raise ValueError(f"Class imbalance too extreme for {symbol}: {class_counts.tolist()}")

    sample_weights = np.where(
        y_train_flat == 1,
        1.0 / class_counts[1],
        1.0 / class_counts[0],
    )

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
    )

    # -------------------------
    # Model training
    # -------------------------
    model = DirectionNet(input_dim=len(FEATURE_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCELoss()

    best_state = None
    best_val_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor)
            val_loss = loss_fn(val_preds, y_val_tensor).item()

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"TrainLoss={total_loss:.4f} | "
            f"ValLoss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # -------------------------
    # Validation metrics
    # -------------------------
    model.eval()
    with torch.no_grad():
        val_probs = model(X_val_tensor).numpy().flatten()

    y_val_labels = y_val.flatten().astype(int)

    opt_long_th, th_stats = _optimize_long_threshold(y_val_labels, val_probs)
    val_pred_labels = (val_probs >= opt_long_th).astype(int)

    metrics = {
        "val_accuracy": float(accuracy_score(y_val_labels, val_pred_labels)),
        "val_precision": float(precision_score(y_val_labels, val_pred_labels, zero_division=0)),
        "val_recall": float(recall_score(y_val_labels, val_pred_labels, zero_division=0)),
        "val_f1": float(f1_score(y_val_labels, val_pred_labels, zero_division=0)),
        "val_positive_rate": float(val_pred_labels.mean()),
    }

    print(
        "Validation | "
        f"Acc={metrics['val_accuracy']:.3f} "
        f"Prec={metrics['val_precision']:.3f} "
        f"Rec={metrics['val_recall']:.3f} "
        f"F1={metrics['val_f1']:.3f}"
    )
    print(
        f"Optimized long threshold={opt_long_th:.2f} | "
        f"Prec={th_stats.get('precision', 0.0):.3f} "
        f"TradeRate={th_stats.get('trade_rate', 0.0):.3f}"
    )

    # -------------------------
    # SAVE
    # -------------------------
    folder = f"models/{symbol.replace('/', '_')}"
    os.makedirs(folder, exist_ok=True)

    torch.save(model.state_dict(), f"{folder}/model.pt")
    joblib.dump(scaler, f"{folder}/scaler.save")

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "symbol": symbol,
        "feature_columns": FEATURE_COLUMNS,
        "horizon": HORIZON,
        "stop_atr_mult": STOP_ATR_MULT,
        "take_atr_mult": TAKE_ATR_MULT,
        "timeframe": TIMEFRAME,
        "optimized_long_threshold": float(opt_long_th),
        "threshold_optimization": th_stats,
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }

    with open(f"{folder}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved {MODEL_NAME} {MODEL_VERSION} to {folder}")


def main():
    for sym in SYMBOLS:
        try:
            train_for_symbol(sym)
        except Exception as e:
            print(f"❌ Failed for {sym}: {e}")


if __name__ == "__main__":
    main()
