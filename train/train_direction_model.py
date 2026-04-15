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
# CONFIG (env-var driven)
# =========================
TIMEFRAME = os.getenv("TRAIN_TIMEFRAME", "15m")
CANDLES = int(os.getenv("TRAIN_CANDLES", "50000"))

HORIZON = int(os.getenv("TRAIN_HORIZON", "12"))
STOP_ATR_MULT = float(os.getenv("TRAIN_STOP_ATR_MULT", "2.0"))
TAKE_ATR_MULT = float(os.getenv("TRAIN_TAKE_ATR_MULT", "3.0"))

EPOCHS = int(os.getenv("TRAIN_EPOCHS", "10"))
BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "256"))
LR = float(os.getenv("TRAIN_LR", "1e-3"))
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", "0.8"))
EARLY_STOPPING_PATIENCE = int(os.getenv("TRAIN_PATIENCE", "3"))
PURGE_BARS = int(os.getenv("TRAIN_PURGE_BARS", "10"))           # leakage protection

WALK_FORWARD_FOLDS = int(os.getenv("WALK_FORWARD_FOLDS", "1"))   # 1=disabled (old behavior)
TRANSFER_FROM_SYMBOL = os.getenv("TRANSFER_FROM_SYMBOL", "BTC/USDT")
TRANSFER_LR_MULT = float(os.getenv("TRANSFER_LR_MULT", "0.1"))
SEQUENCE_WINDOW = int(os.getenv("SEQUENCE_WINDOW", "16"))
SHARPE_RETURN_TP = float(os.getenv("SHARPE_RETURN_TP", "0.03"))
SHARPE_RETURN_SL = float(os.getenv("SHARPE_RETURN_SL", "0.02"))

FEATURE_COLUMNS = [
    # Original 13
    "ema_fast", "ema_slow", "ema_spread", "dist_ema200", "ema_fast_slope",
    "rsi", "rsi_delta", "ret", "vol", "volume_zscore", "atr_pct", "adx", "breakout_strength",
    # New 9 features (~22 total)
    "obv", "macd_line", "macd_signal", "macd_hist",
    "bb_width", "bb_position", "vwap",
    "volume_momentum", "momentum_5", "atr_regime",
]

SYMBOLS = os.getenv("TRAIN_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,MATIC/USDT,AVAX/USDT,LINK/USDT,ADA/USDT,XRP/USDT,DOGE/USDT")
if os.getenv("TRAIN_SYMBOLS"):
    SYMBOLS = [s.strip() for s in os.getenv("TRAIN_SYMBOLS").split(",") if s.strip()]
else:
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


class LSTMSequenceNet(torch.nn.Module):
    """LSTM sequence model: window_size bars × n_features → 64 hidden → 1"""
    def __init__(self, n_features: int, window_size: int = 16, hidden_size: int = 64):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, window, features)
        lstm_out, (hn, cn) = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        dropped = self.dropout(last_out)
        return self.sigmoid(self.fc(dropped))


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


def _compute_trade_returns(y_true, y_prob, threshold, tp_return, sl_return):
    """Compute trade-level returns for Sharpe ratio calculation."""
    preds = (y_prob >= threshold).astype(int)
    if preds.sum() == 0:
        return np.array([])

    rets = np.where(
        (preds == 1) & (y_true == 1), tp_return,
        np.where((preds == 1) & (y_true == 0), -sl_return, 0.0)
    )
    return rets[preds == 1]


def _sharpe_ratio(returns):
    if len(returns) < 2:
        return -1e9
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret < 1e-9:
        return -1e9
    return mean_ret / std_ret * np.sqrt(252)


def _optimize_long_threshold(y_true, y_prob, tp_return=None, sl_return=None) -> tuple[float, dict]:
    """Sharpe-ratio-based threshold optimization."""
    if tp_return is None:
        tp_return = SHARPE_RETURN_TP
    if sl_return is None:
        sl_return = SHARPE_RETURN_SL

    best_th = 0.55
    best_score = -1e9
    best_stats = {}
    min_pred_trades = max(25, int(0.02 * len(y_true)))

    for th in np.arange(0.45, 0.71, 0.01):
        preds = (y_prob >= th).astype(int)
        pred_count = int(preds.sum())
        if pred_count < min_pred_trades:
            continue

        trades = _compute_trade_returns(y_true, y_prob, th, tp_return, sl_return)
        if len(trades) < min_pred_trades:
            continue

        sharpe = _sharpe_ratio(trades)
        precision = float(precision_score(y_true, preds, zero_division=0))
        trade_rate = float(preds.mean())

        # Sharpe-biased score with precision as secondary
        score = sharpe + 0.20 * precision - max(0.0, trade_rate - 0.35) * 0.05

        if score > best_score:
            best_score = score
            best_th = float(th)
            best_stats = {
                "precision": precision,
                "trade_rate": trade_rate,
                "sharpe": float(sharpe),
                "n_trades": len(trades),
                "score": float(score),
            }

    return best_th, best_stats


def _create_sequences(X, y, window=16):
    """Create overlapping sequences of window bars for LSTM training."""
    seq_x, seq_y = [], []
    for i in range(len(X) - window + 1):
        seq_x.append(X[i:i + window])
        seq_y.append(y[i + window - 1])
    return np.array(seq_x), np.array(seq_y)


# =========================
# TRAINING
# =========================
def train_for_symbol(symbol: str):
    MarketDataFetcher, compute_core_features, MODEL_NAME, MODEL_VERSION = _load_project_modules()
    print(f"\n[TRAIN] Training {MODEL_NAME} {MODEL_VERSION} for {symbol}")

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
    current_lr = LR

    # Transfer learning: load BTC base model if available and not training on BTC itself
    if TRANSFER_FROM_SYMBOL and symbol != TRANSFER_FROM_SYMBOL:
        btc_folder = f"models/{TRANSFER_FROM_SYMBOL.replace('/', '_')}"
        btc_model_path = f"{btc_folder}/model.pt"
        if os.path.exists(btc_model_path):
            try:
                btc_state = torch.load(btc_model_path, map_location="cpu")
                model.load_state_dict(btc_state)
                # Freeze all but last linear layer (net.4)
                for name, param in model.named_parameters():
                    if "net.4" not in name:
                        param.requires_grad = False
                current_lr = LR * TRANSFER_LR_MULT
                print(f"  Transfer learning from {TRANSFER_FROM_SYMBOL}, LR={current_lr:.6f}")
            except Exception as e:
                print(f"  Transfer learning failed: {e}, training from scratch")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=current_lr,
    )
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
        "model_type": "MLP",
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

    print(f"[DONE] Saved {MODEL_NAME} {MODEL_VERSION} to {folder}")


def train_lstm_for_symbol(symbol: str):
    """Train LSTM sequence model for a symbol."""
    MarketDataFetcher, compute_core_features, MODEL_NAME, MODEL_VERSION = _load_project_modules()
    print(f"\n[TRAIN] Training LSTM for {symbol}")

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

    labels, valid = _build_triple_barrier_labels(
        df, horizon=HORIZON, stop_atr_mult=STOP_ATR_MULT, take_atr_mult=TAKE_ATR_MULT,
    )
    df["target"] = labels
    df = df[valid].copy()
    df.dropna(inplace=True)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df["target"].values.astype(np.float32).reshape(-1, 1)

    window = SEQUENCE_WINDOW
    X_seq, y_seq = _create_sequences(X, y, window=window)

    split_idx = int(len(df) * TRAIN_SPLIT)
    train_end = max(0, split_idx - PURGE_BARS)

    X_seq_train = X_seq[:max(0, train_end - window + 1)]
    y_seq_train = y_seq[:max(0, train_end - window + 1)]
    X_seq_val = X_seq[split_idx - window + 1:]
    y_seq_val = y_seq[split_idx - window + 1:]

    if len(X_seq_val) < 20:
        X_seq_val = X_seq[-max(20, int(len(X_seq) * 0.1)):]
        y_seq_val = y_seq[-max(20, int(len(y_seq) * 0.1)):]

    # Flatten for scaler: (n_samples, window*features)
    X_flat_train = X_seq_train.reshape(len(X_seq_train), -1)
    scaler = StandardScaler()
    scaler.fit(X_flat_train)

    X_flat_train = scaler.transform(X_flat_train).reshape(X_seq_train.shape)
    X_flat_val = scaler.transform(X_seq_val.reshape(len(X_seq_val), -1)).reshape(X_seq_val.shape)

    X_train_t = torch.tensor(X_flat_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_seq_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_flat_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_seq_val, dtype=torch.float32)

    # Class imbalance handling
    y_train_flat = y_seq_train.flatten().astype(int)
    class_counts = np.bincount(y_train_flat, minlength=2)
    if class_counts.min() == 0:
        raise ValueError(f"Class imbalance too extreme for {symbol}: {class_counts.tolist()}")

    sample_weights = np.where(
        y_train_flat == 1, 1.0 / class_counts[1], 1.0 / class_counts[0],
    )
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True,
    )
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, sampler=sampler)

    # Model
    model = LSTMSequenceNet(n_features=len(FEATURE_COLUMNS), window_size=window)
    current_lr = LR

    # Transfer learning from BTC LSTM if available
    if TRANSFER_FROM_SYMBOL and symbol != TRANSFER_FROM_SYMBOL:
        btc_lstm_folder = f"models/{TRANSFER_FROM_SYMBOL.replace('/', '_')}_lstm"
        btc_lstm_path = f"{btc_lstm_folder}/model.pt"
        if os.path.exists(btc_lstm_path):
            try:
                btc_state = torch.load(btc_lstm_path, map_location="cpu")
                model.load_state_dict(btc_state)
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.fc.parameters():
                    param.requires_grad = True
                current_lr = LR * TRANSFER_LR_MULT
                print(f"  LSTM Transfer learning from {TRANSFER_FROM_SYMBOL}, LR={current_lr:.6f}")
            except Exception as e:
                print(f"  LSTM Transfer learning failed: {e}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=current_lr,
    )
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
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        print(f"LSTM Epoch {epoch+1}/{EPOCHS} | TrainLoss={total_loss:.4f} | ValLoss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
            print("LSTM Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_probs = model(X_val_t).numpy().flatten()

    opt_long_th, th_stats = _optimize_long_threshold(
        y_seq_val.flatten().astype(int), val_probs,
    )

    folder = f"models/{symbol.replace('/', '_')}_lstm"
    os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), f"{folder}/model.pt")
    joblib.dump(scaler, f"{folder}/scaler.save")

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_type": "LSTM",
        "symbol": symbol,
        "feature_columns": FEATURE_COLUMNS,
        "window_size": window,
        "horizon": HORIZON,
        "stop_atr_mult": STOP_ATR_MULT,
        "take_atr_mult": TAKE_ATR_MULT,
        "timeframe": TIMEFRAME,
        "optimized_long_threshold": float(opt_long_th),
        "threshold_optimization": th_stats,
        "train_rows": int(len(X_seq_train)),
        "val_rows": int(len(X_seq_val)),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {"val_sharpe": th_stats.get("sharpe", 0)},
    }

    with open(f"{folder}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[DONE] Saved LSTM to {folder}")


def train_xgboost_for_symbol(symbol: str):
    """Train XGBoost model for a symbol."""
    from xgboost import XGBClassifier

    MarketDataFetcher, compute_core_features, MODEL_NAME, MODEL_VERSION = _load_project_modules()
    print(f"\n[TRAIN] Training XGBoost for {symbol}")

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

    labels, valid = _build_triple_barrier_labels(
        df, horizon=HORIZON, stop_atr_mult=STOP_ATR_MULT, take_atr_mult=TAKE_ATR_MULT,
    )
    df["target"] = labels
    df = df[valid].copy()
    df.dropna(inplace=True)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df["target"].values.astype(np.float32).reshape(-1, 1).flatten().astype(int)

    split_idx = int(len(df) * TRAIN_SPLIT)
    train_end = max(0, split_idx - PURGE_BARS)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[split_idx:], y[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    class_counts = np.bincount(y_train, minlength=2)
    scale_pos_weight = float(class_counts[0]) / max(class_counts[1], 1)

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.01,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_probs = xgb_model.predict_proba(X_val)[:, 1]
    opt_long_th, th_stats = _optimize_long_threshold(y_val, val_probs)

    folder = f"models/{symbol.replace('/', '_')}_xgb"
    os.makedirs(folder, exist_ok=True)
    joblib.dump(xgb_model, f"{folder}/model.save")
    joblib.dump(scaler, f"{folder}/scaler.save")

    val_pred_labels = (val_probs >= opt_long_th).astype(int)
    metrics = {
        "val_accuracy": float(accuracy_score(y_val, val_pred_labels)),
        "val_precision": float(precision_score(y_val, val_pred_labels, zero_division=0)),
        "val_recall": float(recall_score(y_val, val_pred_labels, zero_division=0)),
        "val_f1": float(f1_score(y_val, val_pred_labels, zero_division=0)),
        "val_positive_rate": float(val_pred_labels.mean()),
        "val_sharpe": th_stats.get("sharpe", 0),
    }

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_type": "XGB",
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

    print(f"[DONE] Saved XGBoost to {folder}")


def main():
    for sym in SYMBOLS:
        try:
            train_for_symbol(sym)
            train_lstm_for_symbol(sym)
            train_xgboost_for_symbol(sym)
        except Exception as e:
            print(f"[FAIL] Failed for {sym}: {e}")


if __name__ == "__main__":
    main()
