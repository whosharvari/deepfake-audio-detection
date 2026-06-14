
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import N_MELS

POOLING_TYPES = {"attentive", "plain"}

LCNN_IN_CHANNELS = {"logmel": 1, "lfcc": 1, "multi": 2}

_RESBLOCK_CHANNELS = (32, 64, 128, 128)  
_ASP_ATTENTION_DIM = 128


def mfm(x: torch.Tensor) -> torch.Tensor:
 
    a, b = x.chunk(2, dim=1)
    return torch.max(a, b)


class ResBlock(nn.Module):
   

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * 2, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.bn1(mfm(self.conv1(x)))
        out = self.bn2(mfm(self.conv2(out)))
        return out + identity


class AttentiveStatsPool(nn.Module):
 

    def __init__(self, in_dim: int, attention_dim: int = _ASP_ATTENTION_DIM) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, attention_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(attention_dim),
            nn.Conv1d(attention_dim, in_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.softmax(self.attention(x), dim=2)  # (B, C, T)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * x.pow(2), dim=2) - mean.pow(2)
        std = torch.sqrt(var.clamp(min=1e-8))
        return torch.cat([mean, std], dim=1)  # (B, 2C)


class StatsPool(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        return torch.cat([mean, std], dim=1)


class LCNN(nn.Module):
  
    def __init__(self, in_channels: int = 2, pooling: str = "attentive", dropout: float = 0.3) -> None:
        super().__init__()
        if in_channels not in (1, 2):
            raise ValueError(f"in_channels must be 1 or 2, got {in_channels}")
        if pooling not in POOLING_TYPES:
            raise ValueError(f"pooling must be one of {sorted(POOLING_TYPES)}, got {pooling!r}")

        self.in_channels = in_channels

        c0, c1, c2, c3 = _RESBLOCK_CHANNELS

        self.stem_conv = nn.Conv2d(in_channels, c0 * 2, kernel_size=5, stride=1, padding=2)
        self.stem_bn = nn.BatchNorm2d(c0)
        self.stem_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block1 = ResBlock(c0, c1, stride=2)
        self.block2 = ResBlock(c1, c2, stride=2)
        self.block3 = ResBlock(c2, c3, stride=2)

        pooled_dim = c3 * 5

        self.pool: nn.Module
        if pooling == "attentive":
            self.pool = AttentiveStatsPool(pooled_dim)
        else:
            self.pool = StatsPool()

        self.head = nn.Sequential(
            nn.Linear(pooled_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def _prepare_input(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if self.in_channels == 2:
            if not isinstance(x, dict):
                raise TypeError("in_channels=2 expects a dict with 'logmel' and 'lfcc' keys")
            logmel, lfcc = x["logmel"], x["lfcc"]
            lfcc_resized = F.interpolate(lfcc, size=logmel.shape[-2:], mode="bilinear", align_corners=False)
            return torch.cat([logmel, lfcc_resized], dim=1)

        if isinstance(x, dict):
            raise TypeError("in_channels=1 expects a single spectrogram tensor, not a dict")
        if x.shape[-2] != N_MELS:
            x = F.interpolate(x, size=(N_MELS, x.shape[-1]), mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns raw logits of shape ``(B, 1)``. Apply ``sigmoid`` for ``P(fake)``."""
        x = self._prepare_input(x)

        x = self.stem_pool(self.stem_bn(mfm(self.stem_conv(x))))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        b, c, h, w = x.shape
        x = x.reshape(b, c * h, w)

        pooled = self.pool(x)
        return self.head(pooled)
