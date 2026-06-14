
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config import CHECKPOINTS_DIR, RANDOM_SEED, REPORTS_DIR, RUNS_DIR
from src.data.dataloaders import DEFAULT_NUM_WORKERS, get_dataloaders, get_device, set_seed
from src.evaluation.metrics import format_metrics_report
from src.models.baseline import MODEL_TYPES, train_baseline
from src.models.lcnn import LCNN, LCNN_IN_CHANNELS, POOLING_TYPES
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.losses import get_loss
from src.training.trainer import Trainer

_CHECKPOINT_METRIC_MODE = {
    "loss": "min",
    "eer": "min",
    "accuracy": "max",
    "precision": "max",
    "recall": "max",
    "f1": "max",
    "roc_auc": "max",
}


def _build_loss(args: argparse.Namespace) -> torch.nn.Module:
    if args.loss == "weighted_bce":
        return get_loss("weighted_bce", pos_weight=args.pos_weight)
    if args.loss == "focal":
        return get_loss("focal", gamma=args.focal_gamma, alpha=args.focal_alpha)
    return get_loss("bce")


def _resolve_resume(args: argparse.Namespace, checkpoint: ModelCheckpoint) -> Path | None:
    if args.resume == "auto":
        return checkpoint.last_path if checkpoint.last_path.exists() else None
    if args.resume is not None:
        return Path(args.resume)
    return None


def _run_and_report(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loaders: dict,
    device: torch.device,
) -> None:
    loss_fn = _build_loss(args)

    mode = _CHECKPOINT_METRIC_MODE[args.checkpoint_metric]
    early_stopping = EarlyStopping(patience=args.patience, mode=mode) if args.patience > 0 else None
    checkpoint = ModelCheckpoint(dirpath=CHECKPOINTS_DIR, name=args.run_name, mode=mode)

    grad_clip_norm = args.grad_clip_norm if args.grad_clip_norm > 0 else None

    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        max_epochs=args.epochs,
        grad_clip_norm=grad_clip_norm,
        use_amp=not args.no_amp,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        checkpoint_metric=args.checkpoint_metric,
        log_dir=RUNS_DIR / args.run_name,
    )

    history = trainer.fit(resume_from=_resolve_resume(args, checkpoint))

    trainer.load_checkpoint(checkpoint.best_path)
    test_loss, test_metrics = trainer.evaluate(loaders["test"])
    print()
    print(format_metrics_report(test_metrics, title=f"{args.run_name} - test (best checkpoint)"))

    report = {
        "run_name": args.run_name,
        "args": {key: value for key, value in vars(args).items() if key != "func"},
        "history": history,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{args.run_name}_results.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report to {report_path}")


def run_lcnn(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}  run_name: {args.run_name}")

    loaders = get_dataloaders(
        feature_type=args.feature_type,
        source_datasets=args.source_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = LCNN(in_channels=LCNN_IN_CHANNELS[args.feature_type], pooling=args.pooling, dropout=args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    _run_and_report(args, model, optimizer, scheduler, loaders, device)


def run_baseline(args: argparse.Namespace) -> None:
    save_path = None if args.no_save else CHECKPOINTS_DIR / f"baseline_{args.model_type}.joblib"
    report_path = None if args.no_report else REPORTS_DIR / f"baseline_{args.model_type}_results.json"
    train_baseline(
        model_type=args.model_type,
        source_datasets=args.source_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_path=save_path,
        report_path=report_path,
    )


def build_parser() -> argparse.ArgumentParser:
    
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source-datasets", nargs="+", default=None,
                         help="source_dataset values to train/eval on (default: for-norm).")
    common.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    common.add_argument("--seed", type=int, default=RANDOM_SEED)

    training_common = argparse.ArgumentParser(add_help=False, parents=[common])
    training_common.add_argument("--dropout", type=float, default=0.3)
    training_common.add_argument("--epochs", type=int, default=50)
    training_common.add_argument("--weight-decay", type=float, default=1e-4)
    training_common.add_argument("--grad-clip-norm", type=float, default=5.0,
                                  help="Max gradient norm; <= 0 disables clipping.")
    training_common.add_argument("--loss", choices=("bce", "weighted_bce", "focal"), default="bce")
    training_common.add_argument("--pos-weight", type=float, default=None, help="For --loss weighted_bce.")
    training_common.add_argument("--focal-gamma", type=float, default=2.0, help="For --loss focal.")
    training_common.add_argument("--focal-alpha", type=float, default=0.25, help="For --loss focal.")
    training_common.add_argument("--patience", type=int, default=10,
                                  help="Early-stopping patience in epochs; <= 0 disables early stopping.")
    training_common.add_argument("--checkpoint-metric", choices=sorted(_CHECKPOINT_METRIC_MODE), default="eer")
    training_common.add_argument("--no-amp", action="store_true", help="Disable AMP mixed precision.")
    training_common.add_argument("--run-name", default=None, help="Checkpoint/log/report name.")
    training_common.add_argument("--resume", default=None,
                                  help="'auto' to resume from <run_name>_last.pt if present, "
                                       "or an explicit checkpoint path. Use the same --epochs as the "
                                       "original run: the cosine scheduler's T_max is restored from "
                                       "the checkpoint and will override a changed --epochs.")

    parser = argparse.ArgumentParser(description="Train Models 0/1 for deepfake audio detection.")
    subparsers = parser.add_subparsers(dest="model", required=True)

    p_baseline = subparsers.add_parser("baseline", parents=[common], help="Model 0: LogMel/LFCC stats + XGBoost/RF.")
    p_baseline.add_argument("--batch-size", type=int, default=64)
    p_baseline.add_argument("--model-type", choices=sorted(MODEL_TYPES), default="xgboost")
    p_baseline.add_argument("--no-save", action="store_true", help="Skip saving the fitted model.")
    p_baseline.add_argument("--no-report", action="store_true", help="Skip writing the JSON metrics report.")
    p_baseline.set_defaults(func=run_baseline)

    p_lcnn = subparsers.add_parser("lcnn", parents=[training_common], help="Model 1 (primary): LCNN.")
    p_lcnn.add_argument("--batch-size", type=int, default=64)
    p_lcnn.add_argument("--lr", type=float, default=1e-3)
    p_lcnn.add_argument("--feature-type", choices=("multi", "logmel", "lfcc"), default="multi",
                         help="'multi' -> 2-channel LogMel+LFCC (in_channels=2); "
                              "'logmel'/'lfcc' -> single-feature ablation (in_channels=1).")
    p_lcnn.add_argument("--pooling", choices=sorted(POOLING_TYPES), default="attentive")
    p_lcnn.set_defaults(func=run_lcnn)

    return parser


_RUN_NAME_DEFAULTS = {
    "lcnn": lambda args: f"lcnn_{args.feature_type}_{args.pooling}",
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "run_name", None) is None and args.model in _RUN_NAME_DEFAULTS:
        args.run_name = _RUN_NAME_DEFAULTS[args.model](args)
    args.func(args)


if __name__ == "__main__":
    main()
