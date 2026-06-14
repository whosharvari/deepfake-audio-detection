

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T

from src.config import (
    HOP_LENGTH,
    N_FFT,
    N_LFCC,
    N_LFCC_FILTERS,
    N_MELS,
    SAMPLE_RATE,
    WIN_LENGTH,
)

FEATURE_TYPES = {"logmel", "lfcc", "multi"}


class LogMelSpectrogramExtractor(nn.Module):

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
        n_mels: int = N_MELS,
        top_db: float = 80.0,
    ) -> None:
        super().__init__()
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            center=True,
        )
      
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=top_db)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        power_mel = self.mel_spectrogram(waveform)
        return self.amplitude_to_db(power_mel)


class LFCCExtractor(nn.Module):
    """Linear-Frequency Cepstral Coefficients: ``(1, num_samples) -> (1, n_lfcc, n_frames)``."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_filter: int = N_LFCC_FILTERS,
        n_lfcc: int = N_LFCC,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
        f_min: float = 0.0,
        f_max: float | None = None,
    ) -> None:
        super().__init__()
        self.lfcc = T.LFCC(
            sample_rate=sample_rate,
            n_filter=n_filter,
            f_min=f_min,
            f_max=f_max if f_max is not None else sample_rate / 2,
            n_lfcc=n_lfcc,
            dct_type=2,
            norm="ortho",
            log_lf=True, 
            speckwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "win_length": win_length,
                "power": 2.0,
                "center": True,
            },
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.lfcc(waveform)


class MultiFeatureExtractor(nn.Module):
   

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.logmel = LogMelSpectrogramExtractor(**kwargs.get("logmel_kwargs", {}))
        self.lfcc = LFCCExtractor(**kwargs.get("lfcc_kwargs", {}))

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logmel": self.logmel(waveform), "lfcc": self.lfcc(waveform)}


_FEATURE_EXTRACTOR_CLASSES: dict[str, type[nn.Module]] = {
    "logmel": LogMelSpectrogramExtractor,
    "lfcc": LFCCExtractor,
    "multi": MultiFeatureExtractor,
}


def get_feature_extractor(feature_type: str, **kwargs) -> nn.Module:
   
    if feature_type not in _FEATURE_EXTRACTOR_CLASSES:
        raise ValueError(f"feature_type must be one of {sorted(FEATURE_TYPES)}, got {feature_type!r}")
    return _FEATURE_EXTRACTOR_CLASSES[feature_type](**kwargs)
