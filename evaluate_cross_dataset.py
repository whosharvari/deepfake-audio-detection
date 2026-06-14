

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import CHECKPOINTS_DIR, REPORTS_DIR
from src.data.dataloaders import DEFAULT_NUM_WORKERS, build_dataloader, get_device, set_seed
from src.data.dataset import DeepfakeAudioDataset
from src.evaluation.metrics import format_metrics_report
from src.models.lcnn import LCNN, LCNN_IN_CHANNELS, POOLING_TYPES
from src.training.losses import get_loss
from src.training.trainer import Trainer

_EXPERIMENT_DATASETS = ["for-norm", "for-2sec", "for-rerec"]


def _flatten_metrics(source_dataset: str, n_samples: int, loss: float, metrics: dict) -> dict:
    cm = metrics["confusion_matrix"]
    pca = metrics["per_class_accuracy"]
    return {
        "source_dataset": source_dataset,
        "n_samples": n_samples,
        "loss": loss,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "eer": metrics["eer"],
        "eer_threshold": metrics["eer_threshold"],
        "acc_real": pca.get("real", float("nan")),
        "acc_fake": pca.get("fake", float("nan")),
        "tn": cm[0][0],
        "fp": cm[0][1],
        "fn": cm[1][0],
        "tp": cm[1][1],
    }


def _build_test_loader(source_dataset: str, feature_type: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = DeepfakeAudioDataset(
        split="test", source_datasets=[source_dataset], feature_type=feature_type, augment=False,
    )
    return build_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiments A/B/C: cross-dataset anti-spoofing evaluation.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "lcnn_multi_attentive_v1_best.pt")
    parser.add_argument("--feature-type", choices=("multi", "logmel", "lfcc"), default="multi")
    parser.add_argument("--pooling", choices=sorted(POOLING_TYPES), default="attentive")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--datasets", nargs="+", default=_EXPERIMENT_DATASETS)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "cross_dataset_results.csv")
    args = parser.parse_args()

    set_seed()
    device = get_device()
    print(f"device: {device}  checkpoint: {args.checkpoint}")

    model = LCNN(in_channels=LCNN_IN_CHANNELS[args.feature_type], pooling=args.pooling)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = get_loss("bce")

    first_loader = _build_test_loader(args.datasets[0], args.feature_type, args.batch_size, args.num_workers)
    trainer = Trainer(
        model=model, train_loader=first_loader, val_loader=first_loader,
        loss_fn=loss_fn, optimizer=optimizer, device=device, use_amp=True, verbose=False,
    )
    trainer.load_checkpoint(args.checkpoint)

    rows = []
    for i, source_dataset in enumerate(args.datasets):
        loader = first_loader if i == 0 else _build_test_loader(
            source_dataset, args.feature_type, args.batch_size, args.num_workers
        )
        loss, metrics = trainer.evaluate(loader)
        n_samples = len(loader.dataset)
        print()
        print(format_metrics_report(metrics, title=f"{source_dataset} (n={n_samples})"))
        rows.append(_flatten_metrics(source_dataset, n_samples, loss, metrics))

    df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved cross-dataset results to {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
