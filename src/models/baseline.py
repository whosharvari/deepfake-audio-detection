
from __future__ import annotations

import os



os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CHECKPOINTS_DIR, PRIMARY_SOURCE_DATASETS, RANDOM_SEED, REPORTS_DIR
from src.data.dataloaders import DEFAULT_NUM_WORKERS, build_dataloader, set_seed
from src.data.dataset import DeepfakeAudioDataset
from src.data.manifest import load_manifest
from src.evaluation.metrics import compute_metrics, format_metrics_report

N_LOGMEL_BINS = 80
N_LFCC_BINS = 40
N_STATS = 4  # mean, std, min, max
FEATURE_DIM = (N_LOGMEL_BINS + N_LFCC_BINS) * N_STATS  

MODEL_TYPES = {"xgboost", "rf"}


def _summary_stats_batch(spec: torch.Tensor) -> torch.Tensor:
  
    spec = spec.squeeze(1)  
    stats = torch.stack(
        [spec.mean(dim=2), spec.std(dim=2), spec.amin(dim=2), spec.amax(dim=2)],
        dim=2,
    ) 
    return stats.reshape(spec.shape[0], -1)


def extract_features(loader: DataLoader, desc: str = "extracting features") -> tuple[np.ndarray, np.ndarray]:
    """Build the ``(N, 480)`` feature matrix and ``(N,)`` label vector for `loader`.

    `loader` must be built from a ``feature_type="multi"`` dataset, i.e.
    ``batch["spectrogram"] = {"logmel": (B,1,80,T), "lfcc": (B,1,40,T)}``.
    Only ``spectrogram`` and ``label`` are read — ``metadata`` (which carries
    ``duration``) is never touched.
    """
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    for batch in tqdm(loader, desc=desc):
        logmel_stats = _summary_stats_batch(batch["spectrogram"]["logmel"])
        lfcc_stats = _summary_stats_batch(batch["spectrogram"]["lfcc"])
        feats = torch.cat([logmel_stats, lfcc_stats], dim=1)
        feature_chunks.append(feats.numpy())
        label_chunks.append(batch["label"].numpy())

    X = np.concatenate(feature_chunks, axis=0)
    y = np.concatenate(label_chunks, axis=0)
    return X, y


def build_model(model_type: str = "xgboost", **kwargs):
    """Factory for the baseline classifier (``"xgboost"`` or ``"rf"``)."""
    if model_type == "xgboost":
        from xgboost import XGBClassifier

        params = dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        params.update(kwargs)
        return XGBClassifier(**params)

    if model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier

        params = dict(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        params.update(kwargs)
        return RandomForestClassifier(**params)

    raise ValueError(f"model_type must be one of {sorted(MODEL_TYPES)}, got {model_type!r}")


def train_baseline(
    model_type: str = "xgboost",
    source_datasets: list[str] | None = None,
    batch_size: int = 64,
    num_workers: int = DEFAULT_NUM_WORKERS,
    save_path: Path | str | None = None,
    report_path: Path | str | None = None,
    **model_kwargs,
) -> dict:
   
    set_seed(RANDOM_SEED)

    manifest = load_manifest()
    source_datasets = source_datasets or PRIMARY_SOURCE_DATASETS

    datasets = {
        split: DeepfakeAudioDataset(
            split=split,
            manifest=manifest,
            source_datasets=source_datasets,
            feature_type="multi",
            augment=False,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: build_dataloader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for split, ds in datasets.items()
    }

    t0 = time.time()
    X_train, y_train = extract_features(loaders["train"], desc="features[train]")
    X_val, y_val = extract_features(loaders["val"], desc="features[val]")
    X_test, y_test = extract_features(loaders["test"], desc="features[test]")
    extraction_time = time.time() - t0
    print(f"Feature extraction: {extraction_time:.1f}s "
          f"(train={len(y_train)}, val={len(y_val)}, test={len(y_test)}, dim={FEATURE_DIM})")

    model = build_model(model_type, **model_kwargs)

    t0 = time.time()
    if model_type == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training ({model_type}): {train_time:.1f}s")

    results: dict = {
        "model_type": model_type,
        "feature_dim": FEATURE_DIM,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "extraction_time_sec": extraction_time,
        "train_time_sec": train_time,
    }

    for split_name, X, y in (("val", X_val, y_val), ("test", X_test, y_test)):
        y_prob = model.predict_proba(X)[:, 1]
        metrics = compute_metrics(y, y_prob)
        results[split_name] = metrics
        print()
        print(format_metrics_report(metrics, title=f"Baseline ({model_type}) - {split_name}"))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        print(f"\nSaved model to {save_path}")

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved metrics report to {report_path}")

    return {"model": model, **results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model 0 (baseline): LogMel/LFCC stats + XGBoost/RF.")
    parser.add_argument("--model-type", choices=sorted(MODEL_TYPES), default="xgboost")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--no-save", action="store_true", help="Skip saving the fitted model.")
    parser.add_argument("--no-report", action="store_true", help="Skip writing the JSON metrics report.")
    args = parser.parse_args()

    save_path = None if args.no_save else CHECKPOINTS_DIR / f"baseline_{args.model_type}.joblib"
    report_path = None if args.no_report else REPORTS_DIR / f"baseline_{args.model_type}_results.json"

    train_baseline(
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_path=save_path,
        report_path=report_path,
    )


if __name__ == "__main__":
    main()
