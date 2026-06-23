"""Phase-3 bridge to Phase-2 predictors.

This module prepares reusable one-step prediction artifacts so the RL stage can
consume predicted next-interval load without refitting a predictor inside every
rollout or evaluation job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phase2.predictors import BaseTMPredictor, build_predictor, compute_prediction_metrics, evaluate_predictor_sequence


@dataclass
class PredictorArtifact:
    dataset_key: str
    predictor_name: str
    predictions: np.ndarray
    prediction_available: np.ndarray
    sigma_od: np.ndarray
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    metadata: dict[str, Any]

    def scaled_prediction(self, timestep: int, scale: float = 1.0) -> np.ndarray:
        t = int(timestep)
        if t < 0 or t >= self.predictions.shape[0] or not bool(self.prediction_available[t]):
            raise KeyError(f"Prediction unavailable for timestep {t}")
        return np.asarray(self.predictions[t], dtype=float) * float(scale)


DEFAULT_PREDICTOR_CFG = {
    "predictor": "ensemble",
    "predictor_window": 6,
    "predictor_alpha": 1e-2,
    "season_lag": None,
    "lstm_hidden_dim": 64,
    "lstm_layers": 1,
    "lstm_epochs": 24,
    "lstm_batch_size": 32,
    "lstm_lr": 1e-3,
    "lstm_patience": 5,
    "lstm_model_type": "lstm",
}


def _season_lag_for_dataset(dataset_key: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    return 96 if str(dataset_key).lower() == "geant" else 288


def _artifact_file(output_dir: Path, dataset_key: str, predictor_name: str) -> Path:
    return Path(output_dir) / dataset_key / f"{predictor_name}_artifact.npz"


def _artifact_metadata_file(output_dir: Path, dataset_key: str, predictor_name: str) -> Path:
    return Path(output_dir) / dataset_key / f"{predictor_name}_artifact.json"


def build_phase2_predictor(dataset, phase2_cfg: dict[str, Any] | None, predictor_name: str | None = None) -> BaseTMPredictor:
    cfg = dict(DEFAULT_PREDICTOR_CFG)
    if isinstance(phase2_cfg, dict):
        cfg.update({k: v for k, v in phase2_cfg.items() if v is not None})
    name = str(predictor_name or cfg.get("predictor", "ensemble"))
    return build_predictor(
        name=name,
        window=int(cfg.get("predictor_window", 6)),
        alpha=float(cfg.get("predictor_alpha", 1e-2)),
        season_lag=_season_lag_for_dataset(dataset.key, cfg.get("season_lag")),
        lstm_hidden_dim=int(cfg.get("lstm_hidden_dim", 64)),
        lstm_layers=int(cfg.get("lstm_layers", 1)),
        lstm_epochs=int(cfg.get("lstm_epochs", 24)),
        lstm_batch_size=int(cfg.get("lstm_batch_size", 32)),
        lstm_lr=float(cfg.get("lstm_lr", 1e-3)),
        lstm_patience=int(cfg.get("lstm_patience", 5)),
        lstm_model_type=str(cfg.get("lstm_model_type", "lstm")),
    )


def prepare_predictor_artifact(
    dataset,
    output_dir: Path | str,
    phase2_cfg: dict[str, Any] | None,
    *,
    predictor_name: str | None = None,
    seed: int = 42,
    force: bool = False,
) -> Path:
    predictor = build_phase2_predictor(dataset, phase2_cfg, predictor_name=predictor_name)
    predictor_name = str(predictor.name)
    out_path = _artifact_file(Path(output_dir), dataset.key, predictor_name)
    meta_path = _artifact_metadata_file(Path(output_dir), dataset.key, predictor_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and meta_path.exists() and not force:
        try:
            existing = np.load(out_path, allow_pickle=True)
            pred = np.asarray(existing["predictions"])
            if pred.shape[0] == dataset.tm.shape[0] and pred.shape[1] == dataset.tm.shape[1]:
                return out_path
        except Exception:
            pass

    tm = np.asarray(dataset.tm, dtype=float)
    train_end = int(dataset.split["train_end"])
    val_end = int(dataset.split["val_end"])

    tm_train = tm[:train_end]
    tm_val = tm[train_end:val_end]
    predictor.fit(tm_train, tm_val, seed=int(seed))

    pred_all = np.zeros_like(tm, dtype=np.float32)
    pred_mask = np.zeros(tm.shape[0], dtype=bool)
    for t_idx in range(1, tm.shape[0]):
        pred_all[t_idx] = predictor.predict_next(tm[:t_idx]).astype(np.float32)
        pred_mask[t_idx] = True

    val_idx = range(train_end, val_end)
    test_idx = range(dataset.split["test_start"], tm.shape[0])
    pred_val, actual_val, _ = evaluate_predictor_sequence(predictor, tm, val_idx)
    pred_test, actual_test, _ = evaluate_predictor_sequence(predictor, tm, test_idx)
    val_metrics = compute_prediction_metrics(pred_val, actual_val)
    test_metrics = compute_prediction_metrics(pred_test, actual_test)

    if pred_val.shape[0] > 0:
        sigma_od = np.std(actual_val - pred_val, axis=0).astype(np.float32)
    else:
        sigma_od = np.zeros(tm.shape[1], dtype=np.float32)

    meta = {
        "dataset_key": dataset.key,
        "predictor_name": predictor_name,
        "seed": int(seed),
        "train_end": train_end,
        "val_end": val_end,
        "test_start": int(dataset.split["test_start"]),
        "required_history": int(predictor.required_history()),
        "val_metrics": {
            "mae": float(val_metrics.mae),
            "rmse": float(val_metrics.rmse),
            "smape": float(val_metrics.smape),
        },
        "test_metrics": {
            "mae": float(test_metrics.mae),
            "rmse": float(test_metrics.rmse),
            "smape": float(test_metrics.smape),
        },
    }

    np.savez_compressed(
        out_path,
        predictions=pred_all,
        prediction_available=pred_mask,
        sigma_od=sigma_od,
        metadata_json=np.asarray(json.dumps(meta)),
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_path


def load_predictor_artifact(path: Path | str) -> PredictorArtifact:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(
            f"Phase-2 predictor artifact not found: {src}. Run scripts/run_phase3_train.sh first, or use fallback mode explicitly."
        )

    payload = np.load(src, allow_pickle=True)
    metadata = json.loads(str(payload["metadata_json"].item())) if "metadata_json" in payload else {}
    return PredictorArtifact(
        dataset_key=str(metadata.get("dataset_key", src.parent.name)),
        predictor_name=str(metadata.get("predictor_name", src.stem.replace("_artifact", ""))),
        predictions=np.asarray(payload["predictions"], dtype=float),
        prediction_available=np.asarray(payload["prediction_available"], dtype=bool),
        sigma_od=np.asarray(payload["sigma_od"], dtype=float),
        val_metrics=dict(metadata.get("val_metrics", {})),
        test_metrics=dict(metadata.get("test_metrics", {})),
        metadata=metadata,
    )
