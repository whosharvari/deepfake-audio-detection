

from __future__ import annotations

import math
import random
from functools import lru_cache
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from src.config import SAMPLE_RATE, TARGET_LENGTH_SAMPLES

VALID_MODES = {"train", "eval"}


@lru_cache(maxsize=None)
def _get_resampler(orig_sr: int, target_sr: int) -> torchaudio.transforms.Resample:
  
    return torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
  
    if waveform.shape[0] > 1:
        return waveform.mean(dim=0, keepdim=True)
    return waveform


def load_audio(filepath: str | Path, target_sr: int = SAMPLE_RATE) -> torch.Tensor:

    data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)  # (channels, num_samples)
    waveform = to_mono(waveform)

    if sr != target_sr:
        waveform = _get_resampler(sr, target_sr)(waveform)

    return waveform


def fix_length(
    waveform: torch.Tensor,
    target_length: int = TARGET_LENGTH_SAMPLES,
    mode: str = "train",
) -> torch.Tensor:
   
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    num_samples = waveform.shape[-1]
    if num_samples <= 0:
        raise ValueError(
            "Cannot fix length of an empty waveform (0 samples). "
            "This file should have been filtered out via the manifest's "
            "duration/sample_rate validity check."
        )

    if num_samples == target_length:
        return waveform

    if num_samples > target_length:
        max_start = num_samples - target_length
        start = random.randint(0, max_start) if mode == "train" else max_start // 2
        return waveform[..., start : start + target_length]

    # num_samples < target_length -> wrap-pad (tile, no zeros introduced)
    n_repeats = math.ceil(target_length / num_samples)
    tiled = waveform.repeat(1, n_repeats)
    max_start = tiled.shape[-1] - target_length
    start = random.randint(0, max_start) if mode == "train" else 0
    return tiled[..., start : start + target_length]


def preprocess_waveform(
    filepath: str | Path,
    mode: str = "train",
    target_length: int = TARGET_LENGTH_SAMPLES,
    target_sr: int = SAMPLE_RATE,
) -> torch.Tensor:
   
    waveform = load_audio(filepath, target_sr=target_sr)
    return fix_length(waveform, target_length=target_length, mode=mode)
