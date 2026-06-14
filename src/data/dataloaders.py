
from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import MANIFEST_PATH, RANDOM_SEED
from src.data.dataset import DeepfakeAudioDataset
from src.data.manifest import load_manifest

DEFAULT_NUM_WORKERS = max(1, (os.cpu_count() or 2) - 2)


def set_seed(seed: int = RANDOM_SEED) -> None:
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
   
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_device() -> torch.device:
    
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloader(
    dataset: DeepfakeAudioDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = DEFAULT_NUM_WORKERS,
    generator: torch.Generator | None = None,
    drop_last: bool = False,
) -> DataLoader:
   
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        pin_memory=False,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def get_dataloaders(
    feature_type: str = "logmel",
    source_datasets: list[str] | None = None,
    batch_size: int = 32,
    num_workers: int = DEFAULT_NUM_WORKERS,
    manifest_path=MANIFEST_PATH,
    seed: int = RANDOM_SEED,
    feature_kwargs: dict | None = None,
) -> dict[str, DataLoader]:
 
    manifest: pd.DataFrame = load_manifest(manifest_path)

    datasets = {
        split: DeepfakeAudioDataset(
            split=split,
            manifest=manifest,
            source_datasets=source_datasets,
            feature_type=feature_type,
            feature_kwargs=feature_kwargs,
            augment=(split == "train"),
        )
        for split in ("train", "val", "test")
    }

    train_generator = torch.Generator().manual_seed(seed)

    return {
        "train": build_dataloader(
            datasets["train"], batch_size, shuffle=True, num_workers=num_workers,
            generator=train_generator, drop_last=True,
        ),
        "val": build_dataloader(
            datasets["val"], batch_size, shuffle=False, num_workers=num_workers, drop_last=False,
        ),
        "test": build_dataloader(
            datasets["test"], batch_size, shuffle=False, num_workers=num_workers, drop_last=False,
        ),
    }
