
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config import CHECKPOINTS_DIR, REPORTS_DIR
from src.data.dataloaders import get_dataloaders, get_device, set_seed
from src.evaluation.metrics import format_metrics_report
from src.models.lcnn import LCNN, LCNN_IN_CHANNELS, POOLING_TYPES
from src.training.losses import get_loss
from src.training.trainer import Trainer

RUN_NAME = "lcnn_multi_attentive_v1"

_THRESHOLD_DEPENDENT_KEYS = ("accuracy", "precision", "recall", "f1", "confusion_matrix", "per_class_accuracy")


def _evaluate_dual_threshold(trainer: Trainer, loader) -> tuple[float, dict, dict]:

    trainer.threshold = 0.5
    loss, metrics_05 = trainer.evaluate(loader)

    trainer.threshold = metrics_05["eer_threshold"]
    _, metrics_eer = trainer.evaluate(loader)
    trainer.threshold = 0.5

    return loss, metrics_05, metrics_eer


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B: evaluate the primary LCNN on val + test.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / f"{RUN_NAME}_best.pt")
    parser.add_argument("--feature-type", choices=("multi", "logmel", "lfcc"), default="multi")
    parser.add_argument("--pooling", choices=sorted(POOLING_TYPES), default="attentive")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "lcnn_primary_results.json")
    args = parser.parse_args()

    set_seed()
    device = get_device()
    print(f"device: {device}  checkpoint: {args.checkpoint}")

    loaders = get_dataloaders(feature_type=args.feature_type, batch_size=args.batch_size)

    model = LCNN(in_channels=LCNN_IN_CHANNELS[args.feature_type], pooling=args.pooling)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = get_loss("bce")

    trainer = Trainer(
        model=model, train_loader=loaders["train"], val_loader=loaders["val"],
        loss_fn=loss_fn, optimizer=optimizer, device=device, use_amp=True, verbose=False,
    )
    trainer.load_checkpoint(args.checkpoint)

    val_loss, val_metrics_05, val_metrics_eer = _evaluate_dual_threshold(trainer, loaders["val"])
    test_loss, test_metrics_05, test_metrics_eer = _evaluate_dual_threshold(trainer, loaders["test"])

    n_val, n_test = len(loaders["val"].dataset), len(loaders["test"].dataset)

    print()
    print(format_metrics_report(val_metrics_05, title=f"{RUN_NAME} - val @ threshold=0.5 (n={n_val})"))
    print()
    print(format_metrics_report(val_metrics_eer, title=f"{RUN_NAME} - val @ eer_threshold={val_metrics_05['eer_threshold']:.4f} (n={n_val})"))
    print()
    print(format_metrics_report(test_metrics_05, title=f"{RUN_NAME} - test @ threshold=0.5 (n={n_test})"))
    print()
    print(format_metrics_report(test_metrics_eer, title=f"{RUN_NAME} - test @ eer_threshold={test_metrics_05['eer_threshold']:.4f} (n={n_test})"))

    train_results_path = REPORTS_DIR / f"{RUN_NAME}_results.json"
    history = None
    if train_results_path.exists():
        history = json.loads(train_results_path.read_text()).get("history")

    val_metrics = dict(val_metrics_05)
    val_metrics["at_eer_threshold"] = {k: val_metrics_eer[k] for k in _THRESHOLD_DEPENDENT_KEYS}

    test_metrics = dict(test_metrics_05)
    test_metrics["at_eer_threshold"] = {k: test_metrics_eer[k] for k in _THRESHOLD_DEPENDENT_KEYS}

    report = {
        "run_name": RUN_NAME,
        "checkpoint": str(args.checkpoint),
        "n_val": n_val,
        "n_test": n_test,
        "val_loss": val_loss,
        "val_metrics": val_metrics,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "history": history,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nSaved primary evaluation report to {args.output}")


if __name__ == "__main__":
    main()
