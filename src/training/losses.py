
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits.view(-1), targets.float().view(-1))


class WeightedBCEWithLogitsLoss(nn.Module):
  

    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits.view(-1), targets.float().view(-1), pos_weight=self.pos_weight
        )


class FocalLoss(nn.Module):
  

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.float().view(-1)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "bce": BCELoss,
    "weighted_bce": WeightedBCEWithLogitsLoss,
    "focal": FocalLoss,
}


def get_loss(name: str, **kwargs) -> nn.Module:
    if name not in LOSS_REGISTRY:
        raise ValueError(f"name must be one of {sorted(LOSS_REGISTRY)}, got {name!r}")
    return LOSS_REGISTRY[name](**kwargs)


def compare_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float | None = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> dict[str, float]:
    losses = {
        "bce": BCELoss(),
        "weighted_bce": WeightedBCEWithLogitsLoss(pos_weight=pos_weight),
        "focal": FocalLoss(gamma=gamma, alpha=alpha),
    }
    return {name: fn(logits, targets).item() for name, fn in losses.items()}
