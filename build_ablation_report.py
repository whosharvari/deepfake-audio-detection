

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.config import REPORTS_DIR
from src.models.lcnn import LCNN

_CONFIGS = [
    ("lcnn_logmel_attentive", "Log-Mel only", "attentive", 1),
    ("lcnn_lfcc_attentive", "LFCC only", "attentive", 1),
    ("lcnn_multi_attentive_v1", "Log-Mel + LFCC", "attentive", 2),
    ("lcnn_multi_plain", "Log-Mel + LFCC", "plain", 2),
]

_KNOWN_TRAINING_TIME = {
    "lcnn_multi_attentive_v1": "2h 41m",
    "lcnn_lfcc_attentive": "1h 14m",
}

_TIME_LOG = {
    "lcnn_logmel_attentive": Path("/tmp/phase_d_lcnn_logmel_attentive.log"),
    "lcnn_multi_plain": Path("/tmp/phase_d_lcnn_multi_plain.log"),
}

_REAL_TIME_RE = re.compile(r"^real\s+(\d+)m([\d.]+)s", re.MULTILINE)


def _training_time(run_name: str) -> str:
    if run_name in _KNOWN_TRAINING_TIME:
        return _KNOWN_TRAINING_TIME[run_name]

    log_path = _TIME_LOG[run_name]
    match = _REAL_TIME_RE.search(log_path.read_text())
    if match is None:
        raise ValueError(f"no 'real' timing line found in {log_path}")

    minutes, seconds = int(match.group(1)), float(match.group(2))
    total_minutes = minutes + seconds / 60
    hours, rem_minutes = divmod(total_minutes, 60)
    return f"{int(hours)}h {round(rem_minutes)}m" if hours else f"{round(rem_minutes)}m"


def main() -> None:
    rows = []
    for run_name, feature_label, pooling, in_channels in _CONFIGS:
        results = json.loads((REPORTS_DIR / f"{run_name}_results.json").read_text())
        m = results["test_metrics"]

        model = LCNN(in_channels=in_channels, pooling=pooling)
        n_params = sum(p.numel() for p in model.parameters())

        rows.append({
            "run_name": run_name,
            "features": feature_label,
            "pooling": pooling,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "eer": m["eer"],
            "n_params": n_params,
            "training_time": _training_time(run_name),
        })

    df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "ablation_results.csv"
    df.to_csv(out_path, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved ablation results to {out_path}")


if __name__ == "__main__":
    main()
