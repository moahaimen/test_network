"""Traffic-matrix predictors for Phase-2 proactive TE experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

EPS = 1e-12


@dataclass
class PredictionMetrics:
    mae: float
    rmse: float
    smape: float


class BaseTMPredictor:
    """Base interface for one-step TM predictors."""

    name: str = "base"

    def required_history(self) -> int:
        return 1

    def fit(self, tm_train: np.ndarray, tm_val: np.ndarray | None = None, seed: int = 42) -> None:
        _ = (tm_train, tm_val, seed)

    def predict_next(self, history: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NaiveLastPredictor(BaseTMPredictor):
    """Persistence baseline: next demand equals the last observed demand."""

    name = "naive_last"

    def predict_next(self, history: np.ndarray) -> np.ndarray:
        hist = np.asarray(history, dtype=float)
        if hist.ndim != 2 or hist.shape[0] == 0:
            raise ValueError("history must be a non-empty [T, |OD|] matrix")
        return np.maximum(hist[-1], 0.0)


class SeasonalNaivePredictor(BaseTMPredictor):
    """
    Seasonal baseline with fixed lag.

    For 5-min Abilene, lag=288 corresponds to one day. If history is shorter
    than the seasonal lag, we fallback to last-value persistence.
    """

    name = "seasonal"

    def __init__(self, season_lag: int = 288):
        if season_lag <= 0:
            raise ValueError("season_lag must be >= 1")
        self.season_lag = int(season_lag)

    def required_history(self) -> int:
        return 1

    def predict_next(self, history: np.ndarray) -> np.ndarray:
        hist = np.asarray(history, dtype=float)
        if hist.ndim != 2 or hist.shape[0] == 0:
            raise ValueError("history must be a non-empty [T, |OD|] matrix")

        if hist.shape[0] > self.season_lag:
            return np.maximum(hist[-self.season_lag], 0.0)
        return np.maximum(hist[-1], 0.0)


class MovingAveragePredictor(BaseTMPredictor):
    """Rolling mean predictor over the most recent window."""

    name = "moving_avg"

    def __init__(self, window: int = 4):
        if window <= 0:
            raise ValueError("window must be >= 1")
        self.window = int(window)

    def required_history(self) -> int:
        return self.window

    def predict_next(self, history: np.ndarray) -> np.ndarray:
        hist = np.asarray(history, dtype=float)
        if hist.ndim != 2 or hist.shape[0] == 0:
            raise ValueError("history must be a non-empty [T, |OD|] matrix")
        w = min(self.window, hist.shape[0])
        return np.maximum(np.mean(hist[-w:], axis=0), 0.0)


class RidgeAutoRegressivePredictor(BaseTMPredictor):
    """
    Per-OD ridge autoregression using only that OD's own lagged values.

    This keeps training cheap even when |OD| is large, while still capturing
    short-term temporal structure beyond naive persistence.
    """

    name = "ar_ridge"

    def __init__(self, window: int = 6, alpha: float = 1e-2):
        if window <= 0:
            raise ValueError("window must be >= 1")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.window = int(window)
        self.alpha = float(alpha)
        self.weights: np.ndarray | None = None  # [|OD|, window]
        self.bias: np.ndarray | None = None  # [|OD|]

    def required_history(self) -> int:
        return self.window

    def fit(self, tm_train: np.ndarray, tm_val: np.ndarray | None = None, seed: int = 42) -> None:
        _ = (tm_val, seed)
        tm = np.asarray(tm_train, dtype=float)
        if tm.ndim != 2 or tm.shape[0] < 2:
            raise ValueError("tm_train must be [T, |OD|] with T >= 2")

        num_steps, num_od = tm.shape
        w = self.window

        weights = np.zeros((num_od, w), dtype=float)
        bias = np.zeros(num_od, dtype=float)

        # We fit one compact model per OD to avoid high-dimensional global regression.
        for od_idx in range(num_od):
            series = tm[:, od_idx]

            x_rows = []
            y_vals = []
            for t in range(w, num_steps):
                # Lag order: [t-1, t-2, ..., t-w]
                lag_vec = series[t - w : t][::-1]
                x_rows.append(lag_vec)
                y_vals.append(series[t])

            if not x_rows:
                # Not enough history: fallback to constant predictor.
                bias[od_idx] = float(series[-1])
                continue

            X = np.asarray(x_rows, dtype=float)
            y = np.asarray(y_vals, dtype=float)
            X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=float)], axis=1)

            reg = self.alpha * np.eye(w + 1, dtype=float)
            reg[-1, -1] = 0.0  # do not regularize intercept

            lhs = X_aug.T @ X_aug + reg
            rhs = X_aug.T @ y

            try:
                beta = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(lhs) @ rhs

            weights[od_idx] = beta[:w]
            bias[od_idx] = float(beta[-1])

        self.weights = weights
        self.bias = bias

    def predict_next(self, history: np.ndarray) -> np.ndarray:
        hist = np.asarray(history, dtype=float)
        if hist.ndim != 2 or hist.shape[0] == 0:
            raise ValueError("history must be a non-empty [T, |OD|] matrix")
        if self.weights is None or self.bias is None:
            raise RuntimeError("predictor must be fitted before predict_next")

        num_od = hist.shape[1]
        if num_od != self.weights.shape[0]:
            raise ValueError("OD dimension mismatch between fit data and history")

        w = self.window
        out = np.zeros(num_od, dtype=float)
        for od_idx in range(num_od):
            series = hist[:, od_idx]
            if series.shape[0] >= w:
                lag_vec = series[-w:][::-1]
            else:
                pad = np.zeros(w, dtype=float)
                pad[: series.shape[0]] = series[::-1]
                lag_vec = pad

            pred = float(np.dot(self.weights[od_idx], lag_vec) + self.bias[od_idx])
            out[od_idx] = max(pred, 0.0)

        return out


class _SeqForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, model_type: str = "lstm"):
        super().__init__()
        m = model_type.strip().lower()
        if m == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        elif m == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown model_type '{model_type}'")
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return self.head(y[:, -1, :])


def _build_supervised_matrices(
    tm: np.ndarray,
    window: int,
    target_start: int,
    target_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    t0 = max(window, int(target_start))
    t1 = min(int(target_end), tm.shape[0])

    for t in range(t0, t1):
        x_rows.append(tm[t - window : t])
        y_rows.append(tm[t])

    if not x_rows:
        d = tm.shape[1] if tm.ndim == 2 else 0
        return np.zeros((0, window, d), dtype=np.float32), np.zeros((0, d), dtype=np.float32)

    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


class LSTMPredictor(BaseTMPredictor):
    """Compact sequence model (LSTM/GRU) for one-step TM forecasting."""

    name = "lstm"

    def __init__(
        self,
        window: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 1,
        model_type: str = "lstm",
        epochs: int = 40,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        patience: int = 6,
        min_delta: float = 1e-4,
    ):
        if window <= 0:
            raise ValueError("window must be >= 1")
        if hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("hidden_dim and num_layers must be >= 1")
        if epochs <= 0:
            raise ValueError("epochs must be >= 1")
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")

        self.window = int(window)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.model_type = str(model_type).lower()
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        self.train_mean: np.ndarray | None = None
        self.train_std: np.ndarray | None = None
        self.model: _SeqForecaster | None = None
        self.fallback_last: np.ndarray | None = None
        self.best_val_loss: float = float("inf")
        self.train_log: list[dict[str, float]] = []

    def required_history(self) -> int:
        return self.window

    def fit(self, tm_train: np.ndarray, tm_val: np.ndarray | None = None, seed: int = 42) -> None:
        train = np.asarray(tm_train, dtype=np.float32)
        if train.ndim != 2 or train.shape[0] < 2:
            raise ValueError("tm_train must be [T, |OD|] with T >= 2")

        if tm_val is None:
            val = np.zeros((0, train.shape[1]), dtype=np.float32)
        else:
            val = np.asarray(tm_val, dtype=np.float32)
            if val.ndim != 2 or val.shape[1] != train.shape[1]:
                raise ValueError("tm_val must have shape [T_val, |OD|] with matching OD dimension")

        self.fallback_last = np.maximum(train[-1].astype(float), 0.0)
        self.train_mean = np.mean(train, axis=0)
        self.train_std = np.std(train, axis=0)
        self.train_std = np.maximum(self.train_std, 1e-6)

        train_n = (train - self.train_mean) / self.train_std
        full_n = np.concatenate([train_n, (val - self.train_mean) / self.train_std], axis=0)

        x_train, y_train = _build_supervised_matrices(train_n, self.window, self.window, train_n.shape[0])
        x_val, y_val = _build_supervised_matrices(full_n, self.window, train_n.shape[0], full_n.shape[0])

        if x_train.shape[0] == 0:
            self.model = None
            return

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        input_dim = int(train.shape[1])
        model = _SeqForecaster(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            model_type=self.model_type,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
            batch_size=min(self.batch_size, max(1, x_train.shape[0])),
            shuffle=True,
            drop_last=False,
        )

        if x_val.shape[0] > 0:
            val_x_t = torch.from_numpy(x_val)
            val_y_t = torch.from_numpy(y_val)
        else:
            val_x_t = None
            val_y_t = None

        best_state = None
        best_val = float("inf")
        stale = 0
        self.train_log = []

        for epoch in range(self.epochs):
            model.train()
            losses = []
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            train_loss = float(np.mean(losses)) if losses else float("inf")

            if val_x_t is not None and val_y_t is not None:
                model.eval()
                with torch.no_grad():
                    val_pred = model(val_x_t)
                    val_loss = float(loss_fn(val_pred, val_y_t).item())
            else:
                val_loss = train_loss

            self.train_log.append({"epoch": float(epoch + 1), "train_loss": train_loss, "val_loss": val_loss})

            if val_loss + self.min_delta < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1

            if stale >= self.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        self.model = model
        self.best_val_loss = best_val

    def _prepare_window(self, hist: np.ndarray) -> np.ndarray:
        if self.train_mean is None or self.train_std is None:
            raise RuntimeError("predictor must be fitted before predict_next")

        h = np.asarray(hist, dtype=np.float32)
        h = (h - self.train_mean) / self.train_std

        if h.shape[0] >= self.window:
            x = h[-self.window :]
        else:
            pad_len = self.window - h.shape[0]
            if h.shape[0] == 0:
                pad = np.zeros((pad_len, self.train_mean.shape[0]), dtype=np.float32)
                x = pad
            else:
                pad = np.repeat(h[:1], pad_len, axis=0)
                x = np.concatenate([pad, h], axis=0)

        return x.astype(np.float32)

    def predict_next(self, history: np.ndarray) -> np.ndarray:
        hist = np.asarray(history, dtype=float)
        if hist.ndim != 2 or hist.shape[0] == 0:
            raise ValueError("history must be a non-empty [T, |OD|] matrix")

        if self.model is None or self.train_mean is None or self.train_std is None:
            if self.fallback_last is None:
                raise RuntimeError("predictor must be fitted before predict_next")
            return self.fallback_last.copy()

        x = self._prepare_window(hist)
        x_t = torch.from_numpy(x[None, :, :])
        with torch.no_grad():
            y = self.model(x_t).cpu().numpy()[0]

        y = y * self.train_std + self.train_mean
        return np.maximum(y.astype(float), 0.0)


def _project_to_simplex(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=float)
    if v.ndim != 1:
        raise ValueError("vec must be 1-D")
    n = v.size
    if n == 0:
        return v

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if rho.size == 0:
        return np.full(n, 1.0 / n, dtype=float)
    idx = int(rho[-1])
    theta = (cssv[idx] - 1.0) / (idx + 1.0)
    w = np.maximum(v - theta, 0.0)
    s = float(np.sum(w))
    if s <= EPS:
        return np.full(n, 1.0 / n, dtype=float)
    return w / s


def _collect_split_predictions(
    predictors: Sequence[BaseTMPredictor],
    tm_train: np.ndarray,
    tm_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if tm_eval.shape[0] == 0:
        d = tm_train.shape[1]
        return np.zeros((0, len(predictors), d), dtype=float), np.zeros((0, d), dtype=float)

    out = np.zeros((tm_eval.shape[0], len(predictors), tm_eval.shape[1]), dtype=float)
    actual = np.zeros_like(tm_eval, dtype=float)

    for i in range(tm_eval.shape[0]):
        history = np.concatenate([tm_train, tm_eval[:i]], axis=0)
        for j, model in enumerate(predictors):
            out[i, j] = model.predict_next(history)
        actual[i] = tm_eval[i]

    return out, actual


class EnsemblePredictor(BaseTMPredictor):
    """
    Weighted ensemble of Seasonal + LSTM + Ridge predictors.

    We fit all component models on train, then learn non-negative weights that
    sum to 1 on validation by minimizing squared error.
    """

    name = "ensemble"

    def __init__(
        self,
        season_lag: int = 288,
        ridge_window: int = 6,
        ridge_alpha: float = 1e-2,
        lstm_window: int = 12,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 1,
        lstm_epochs: int = 40,
        lstm_patience: int = 6,
        lstm_model_type: str = "lstm",
    ):
        self.models: list[BaseTMPredictor] = [
            SeasonalNaivePredictor(season_lag=season_lag),
            LSTMPredictor(
                window=lstm_window,
                hidden_dim=lstm_hidden_dim,
                num_layers=lstm_layers,
                model_type=lstm_model_type,
                epochs=lstm_epochs,
                patience=lstm_patience,
            ),
            RidgeAutoRegressivePredictor(window=ridge_window, alpha=ridge_alpha),
        ]
        self.weights = np.full(len(self.models), 1.0 / len(self.models), dtype=float)

    def required_history(self) -> int:
        return max(model.required_history() for model in self.models)

    def fit(self, tm_train: np.ndarray, tm_val: np.ndarray | None = None, seed: int = 42) -> None:
        train = np.asarray(tm_train, dtype=float)
        if train.ndim != 2 or train.shape[0] == 0:
            raise ValueError("tm_train must be non-empty [T, |OD|]")

        if tm_val is None:
            val = np.zeros((0, train.shape[1]), dtype=float)
        else:
            val = np.asarray(tm_val, dtype=float)

        for idx, model in enumerate(self.models):
            model.fit(train, val, seed=seed + idx)

        if val.shape[0] == 0:
            self.weights = np.full(len(self.models), 1.0 / len(self.models), dtype=float)
            return

        pred_stack, actual = _collect_split_predictions(self.models, train, val)
        # Flatten all OD/timestep targets into one least-squares problem.
        X = pred_stack.transpose(0, 2, 1).reshape(-1, len(self.models))
        y = actual.reshape(-1)

        try:
            w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            w = _project_to_simplex(w)
        except np.linalg.LinAlgError:
            w = np.full(len(self.models), 1.0 / len(self.models), dtype=float)

        self.weights = w

    def predict_next(self, history: np.ndarray) -> np.ndarray:
        preds = [model.predict_next(history) for model in self.models]
        out = np.zeros_like(preds[0], dtype=float)
        for w, pred in zip(self.weights, preds):
            out += float(w) * np.asarray(pred, dtype=float)
        return np.maximum(out, 0.0)


def build_predictor(
    name: str,
    window: int = 6,
    alpha: float = 1e-2,
    season_lag: int = 288,
    lstm_hidden_dim: int = 64,
    lstm_layers: int = 1,
    lstm_epochs: int = 40,
    lstm_batch_size: int = 32,
    lstm_lr: float = 1e-3,
    lstm_patience: int = 6,
    lstm_model_type: str = "lstm",
) -> BaseTMPredictor:
    key = str(name).strip().lower()
    if key in {"naive", "naive_last", "last"}:
        return NaiveLastPredictor()
    if key in {"seasonal", "seasonal_naive"}:
        return SeasonalNaivePredictor(season_lag=season_lag)
    if key in {"ma", "moving_avg", "moving_average"}:
        return MovingAveragePredictor(window=window)
    if key in {"ar", "ar_ridge", "ridge_ar"}:
        return RidgeAutoRegressivePredictor(window=window, alpha=alpha)
    if key in {"lstm", "gru"}:
        return LSTMPredictor(
            window=max(2, window),
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
            model_type=key,
            epochs=lstm_epochs,
            batch_size=lstm_batch_size,
            lr=lstm_lr,
            patience=lstm_patience,
        )
    if key in {"ensemble", "ens"}:
        return EnsemblePredictor(
            season_lag=season_lag,
            ridge_window=window,
            ridge_alpha=alpha,
            lstm_window=max(2, window),
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            lstm_epochs=lstm_epochs,
            lstm_patience=lstm_patience,
            lstm_model_type=lstm_model_type,
        )
    raise ValueError(f"Unknown predictor '{name}'")


def evaluate_predictor_sequence(
    predictor: BaseTMPredictor,
    tm: np.ndarray,
    eval_indices: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate one-step predictions over selected eval indices.

    For each eval index t, prediction uses history tm[:t] only.
    """
    pred_rows = []
    actual_rows = []
    t_rows = []

    arr = np.asarray(tm, dtype=float)
    for t in eval_indices:
        t_idx = int(t)
        if t_idx <= 0 or t_idx >= arr.shape[0]:
            continue
        pred = predictor.predict_next(arr[:t_idx])
        pred_rows.append(pred)
        actual_rows.append(arr[t_idx])
        t_rows.append(t_idx)

    if not pred_rows:
        d = arr.shape[1] if arr.ndim == 2 else 0
        return np.zeros((0, d), dtype=float), np.zeros((0, d), dtype=float), np.zeros(0, dtype=int)

    return (
        np.asarray(pred_rows, dtype=float),
        np.asarray(actual_rows, dtype=float),
        np.asarray(t_rows, dtype=int),
    )


def compute_prediction_metrics(pred: np.ndarray, actual: np.ndarray) -> PredictionMetrics:
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    if p.shape != a.shape:
        raise ValueError("pred and actual must have same shape")

    if p.size == 0:
        return PredictionMetrics(mae=0.0, rmse=0.0, smape=0.0)

    err = p - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))

    denom = np.abs(p) + np.abs(a) + EPS
    smape = float(np.mean(2.0 * np.abs(err) / denom))

    return PredictionMetrics(mae=mae, rmse=rmse, smape=smape)
