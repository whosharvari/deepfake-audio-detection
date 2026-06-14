
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.config import (
    AUDIO_EXTENSIONS,
    DATASET_VERSIONS,
    INVALID_DURATION,
    INVALID_SAMPLE_RATE,
    LABEL_TO_INT,
    MANIFEST_PATH,
    PROJECT_ROOT,
    SPLIT_DIR_TO_LABEL,
)

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = [
    "filepath",
    "label",
    "split",
    "source_dataset",
    "duration",
    "sample_rate",
]


def _iter_audio_files(directory: Path) -> Iterable[Path]:
   
    if not directory.is_dir():
        return
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        if entry.name.startswith("."):
            continue
        if entry.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        yield entry


def _probe_audio(filepath: Path) -> tuple[float, int]:
 
    try:
        info = sf.info(str(filepath))
        if info.samplerate <= 0 or info.frames <= 0:
            raise ValueError(f"non-positive frames/samplerate: {info}")
        duration = info.frames / info.samplerate
        return duration, info.samplerate
    except Exception as exc:  # noqa: BLE001 - intentionally broad: probing must never crash the scan
        logger.warning("Could not read audio header for %s (%s)", filepath, exc)
        return INVALID_DURATION, INVALID_SAMPLE_RATE


def build_manifest(
    dataset_versions: dict[str, Path] | None = None,
    project_root: Path = PROJECT_ROOT,
    show_progress: bool = True,
) -> pd.DataFrame:
   
    dataset_versions = dataset_versions or DATASET_VERSIONS
    rows: list[dict] = []

    for source_dataset, root in dataset_versions.items():
        for split_dir, split_label in SPLIT_DIR_TO_LABEL.items():
            for class_name, label_int in LABEL_TO_INT.items():
                class_dir = root / split_dir / class_name
                if not class_dir.is_dir():
                    logger.warning("Missing expected directory: %s", class_dir)
                    continue

                files = list(_iter_audio_files(class_dir))
                iterator = tqdm(
                    files,
                    desc=f"{source_dataset}/{split_dir}/{class_name}",
                    disable=not show_progress,
                    unit="file",
                )
                for filepath in iterator:
                    duration, sample_rate = _probe_audio(filepath)
                    rows.append(
                        {
                            "filepath": filepath.relative_to(project_root).as_posix(),
                            "label": class_name,
                            "split": split_label,
                            "source_dataset": source_dataset,
                            "duration": duration,
                            "sample_rate": sample_rate,
                        }
                    )

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest = manifest.sort_values(["source_dataset", "split", "label", "filepath"]).reset_index(drop=True)
    # sanity: label_int mapping is validated against LABEL_TO_INT keys at scan time
    assert set(manifest["label"].unique()) <= set(LABEL_TO_INT.keys())
    return manifest


def save_manifest(manifest: pd.DataFrame, path: Path = MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    return path


def load_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def summarize_manifest(manifest: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Total rows: {len(manifest)}")

    counts = manifest.groupby(["source_dataset", "split", "label"]).size().unstack(fill_value=0)
    lines.append("\nFile counts (source_dataset / split x label):")
    lines.append(counts.to_string())

    invalid = manifest[(manifest["duration"] <= 0) | (manifest["sample_rate"] <= 0)]
    lines.append(f"\nInvalid / unreadable files: {len(invalid)}")
    if len(invalid):
        lines.append(invalid[["filepath", "source_dataset", "split", "label"]].to_string(index=False))

    sr_by_dataset = manifest.groupby("source_dataset")["sample_rate"].unique()
    lines.append("\nSample rates observed per source_dataset:")
    lines.append(sr_by_dataset.to_string())

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = build_manifest()
    out_path = save_manifest(df)
    print(f"Saved manifest with {len(df)} rows to {out_path}")
    print()
    print(summarize_manifest(df))
