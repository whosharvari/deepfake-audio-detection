

from __future__ import annotations

import math
from pathlib import Path

import torch

_MODES = {"min", "max"}


class EarlyStopping:
  

    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 0.0) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}, got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_score: float | None = None
        self.best_epoch: int = -1
        self.counter: int = 0
        self.should_stop: bool = False

    def _is_improvement(self, score: float) -> bool:
        if math.isnan(score):
          
            return False
        if self.best_score is None or math.isnan(self.best_score):
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def step(self, score: float, epoch: int) -> bool:
        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def state_dict(self) -> dict:
        return {
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_score = state["best_score"]
        self.best_epoch = state["best_epoch"]
        self.counter = state["counter"]
        self.should_stop = state["should_stop"]


class ModelCheckpoint:
  

    def __init__(self, dirpath: Path | str, name: str = "model", mode: str = "min") -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}, got {mode!r}")
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.mode = mode
        self.best_score: float | None = None

    @property
    def last_path(self) -> Path:
        return self.dirpath / f"{self.name}_last.pt"

    @property
    def best_path(self) -> Path:
        return self.dirpath / f"{self.name}_best.pt"

    def _is_improvement(self, score: float) -> bool:
        if math.isnan(score):
           
            return False
        if self.best_score is None or math.isnan(self.best_score):
            return True
        if self.mode == "min":
            return score < self.best_score
        return score > self.best_score

    def step(self, score: float, state: dict) -> bool:
  
        torch.save(state, self.last_path)
        is_best = self._is_improvement(score)
        if is_best:
            self.best_score = score
            torch.save(state, self.best_path)
        return is_best

    def state_dict(self) -> dict:
        return {"best_score": self.best_score}

    def load_state_dict(self, state: dict) -> None:
        self.best_score = state["best_score"]
