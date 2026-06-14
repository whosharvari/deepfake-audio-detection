from __future__ import annotations

import random

import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional as AF

from src.config import SAMPLE_RATE

_EPS = 1e-10


class RandomGain(nn.Module):
   

    def __init__(self, min_gain_db: float = -6.0, max_gain_db: float = 6.0, p: float = 0.5) -> None:
        super().__init__()
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return waveform
        gain_db = random.uniform(self.min_gain_db, self.max_gain_db)
        factor = 10.0 ** (gain_db / 20.0)
        return torch.clamp(waveform * factor, -1.0, 1.0)


class AdditiveNoise(nn.Module):
   
    def __init__(self, min_snr_db: float = 10.0, max_snr_db: float = 30.0, p: float = 0.5) -> None:
        super().__init__()
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return waveform

        signal_power = waveform.pow(2).mean()
        if signal_power < _EPS:
            return waveform  

        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return torch.clamp(waveform + noise, -1.0, 1.0)


class CompressionSimulation(nn.Module):
    

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        cutoff_rates: tuple[int, ...] = (4000, 6000, 8000, 11025),
        quantization_channels: tuple[int, ...] = (64, 128, 256),
        p: float = 0.3,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.cutoff_rates = cutoff_rates
        self.quantization_channels = quantization_channels
        self.p = p
        
        self._down = nn.ModuleDict(
            {str(rate): T.Resample(orig_freq=sample_rate, new_freq=rate) for rate in cutoff_rates}
        )
        self._up = nn.ModuleDict(
            {str(rate): T.Resample(orig_freq=rate, new_freq=sample_rate) for rate in cutoff_rates}
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return waveform

        original_length = waveform.shape[-1]

        cutoff = random.choice(self.cutoff_rates)
        out = self._down[str(cutoff)](waveform)
        out = self._up[str(cutoff)](out)


        if out.shape[-1] > original_length:
            out = out[..., :original_length]
        elif out.shape[-1] < original_length:
            out = nn.functional.pad(out, (0, original_length - out.shape[-1]))

        channels = random.choice(self.quantization_channels)
        out = torch.clamp(out, -1.0, 1.0)
        encoded = AF.mu_law_encoding(out, channels)
        out = AF.mu_law_decoding(encoded, channels)

        return out


class WaveformAugmentations(nn.Module):
    

    def __init__(
        self,
        gain: RandomGain | None = None,
        noise: AdditiveNoise | None = None,
        compression: CompressionSimulation | None = None,
    ) -> None:
        super().__init__()
        self.gain = gain or RandomGain()
        self.noise = noise or AdditiveNoise()
        self.compression = compression or CompressionSimulation()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = self.gain(waveform)
        waveform = self.noise(waveform)
        waveform = self.compression(waveform)
        return waveform


class SpecAugment(nn.Module):
   

    def __init__(
        self,
        freq_mask_param: int = 8,
        time_mask_param: int = 20,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.p = p
        self.freq_masks = nn.ModuleList([T.FrequencyMasking(freq_mask_param) for _ in range(n_freq_masks)])
        self.time_masks = nn.ModuleList([T.TimeMasking(time_mask_param) for _ in range(n_time_masks)])

    def _apply(self, spec: torch.Tensor) -> torch.Tensor:
        for mask in self.freq_masks:
            spec = mask(spec)
        for mask in self.time_masks:
            spec = mask(spec)
        return spec

    def forward(self, features: torch.Tensor | dict[str, torch.Tensor]):
        if random.random() >= self.p:
            return features

        if isinstance(features, dict):
            return {key: self._apply(value) for key, value in features.items()}
        return self._apply(features)


def build_waveform_augmentations() -> WaveformAugmentations:
   
    return WaveformAugmentations()


def build_spec_augment() -> SpecAugment:
    
    return SpecAugment()
