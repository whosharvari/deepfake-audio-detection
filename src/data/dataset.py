
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import (
    LABEL_TO_INT,
    MANIFEST_PATH,
    PRIMARY_SOURCE_DATASETS,
    PROJECT_ROOT,
    SAMPLE_RATE,
    TARGET_LENGTH_SAMPLES,
)
from src.data.augmentations import build_spec_augment, build_waveform_augmentations
from src.data.features import FEATURE_TYPES, get_feature_extractor
from src.data.manifest import load_manifest
from src.data.preprocessing import preprocess_waveform

logger = logging.getLogger(__name__)

VALID_SPLITS = {"train", "val", "test"}

_PREPROCESS_MODE = {"train": "train", "val": "eval", "test": "eval"}

_NORM_EPS = 1e-6


def _standardize(tensor: torch.Tensor) -> torch.Tensor:
   
    return (tensor - tensor.mean()) / (tensor.std() + _NORM_EPS)


class DeepfakeAudioDataset(Dataset):
   

    def __init__(
        self,
        split: str,
        manifest: pd.DataFrame | None = None,
        manifest_path: Path = MANIFEST_PATH,
        source_datasets: list[str] | None = None,
        feature_type: str = "logmel",
        feature_kwargs: dict | None = None,
        project_root: Path = PROJECT_ROOT,
        target_length: int = TARGET_LENGTH_SAMPLES,
        target_sr: int = SAMPLE_RATE,
        augment: bool = True,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
        if feature_type not in FEATURE_TYPES:
            raise ValueError(f"feature_type must be one of {sorted(FEATURE_TYPES)}, got {feature_type!r}")

        self.split = split
        self.feature_type = feature_type
        self.project_root = Path(project_root)
        self.target_length = target_length
        self.target_sr = target_sr
   
        self.augment = augment and (split == "train")

        df = manifest if manifest is not None else load_manifest(manifest_path)

        df = df[df["split"] == split]

        source_datasets = source_datasets or PRIMARY_SOURCE_DATASETS
        df = df[df["source_dataset"].isin(source_datasets)]

        valid_mask = (df["duration"] > 0) & (df["sample_rate"] > 0)
        n_invalid = int((~valid_mask).sum())
        if n_invalid:
            logger.warning(
                "Dropping %d row(s) with invalid duration/sample_rate from split=%r, source_datasets=%s",
                n_invalid,
                split,
                source_datasets,
            )
        df = df[valid_mask].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No valid rows for split={split!r}, source_datasets={source_datasets}. "
                "Check the manifest and source_datasets filter."
            )

        self.manifest = df

        self.feature_extractor = get_feature_extractor(feature_type, **(feature_kwargs or {}))
        if self.augment:
            self.waveform_augment = build_waveform_augmentations()
            self.spec_augment = build_spec_augment()
        else:
            self.waveform_augment = None
            self.spec_augment = None

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]
        filepath = self.project_root / row["filepath"]

        waveform = preprocess_waveform(
            filepath,
            mode=_PREPROCESS_MODE[self.split],
            target_length=self.target_length,
            target_sr=self.target_sr,
        )

        if self.augment:
            waveform = self.waveform_augment(waveform)

        features = self.feature_extractor(waveform)

        if isinstance(features, dict):
            features = {key: _standardize(value) for key, value in features.items()}
        else:
            features = _standardize(features)

        if self.augment:
            features = self.spec_augment(features)

        label = torch.tensor(LABEL_TO_INT[row["label"]], dtype=torch.long)

        metadata = {
            "filepath": row["filepath"],
            "source_dataset": row["source_dataset"],
            "split": row["split"],
            "duration": float(row["duration"]),
            "sample_rate": int(row["sample_rate"]),
        }

        return {"spectrogram": features, "label": label, "metadata": metadata}
