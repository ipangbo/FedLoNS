"""
Utility functions for reproducibility, device management, and loss helpers.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable

import _compat  # noqa: F401
import numpy as np
import torch
from torch import Tensor


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_torch_device(preferred: str | None = None) -> torch.device:
    """
    Resolve a torch.device given an optional user preference.

    Falls back to CUDA when available, otherwise returns CPU.
    """

    if preferred:
        try:
            device = torch.device(preferred)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available.")
            return device
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid preferred device '{preferred}'") from exc

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gaussian_nll_loss(target: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Compute Gaussian negative log-likelihood loss."""

    var = std.pow(2).clamp_min(1e-6)
    log_term = torch.log(var)
    inv_var = var.reciprocal()
    return 0.5 * ((target - mean).pow(2) * inv_var + log_term + math.log(2.0 * math.pi))


@dataclass
class MetricTracker:
    """Simple meter for accumulating loss values."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)


def weighted_average_state_dict(
    state_dicts: Iterable[tuple[Dict[str, Tensor], float]]
) -> Dict[str, Tensor]:
    """
    Compute the weighted average of model parameters.

    Args:
        state_dicts: Iterable of tuples (state_dict, weight) where weights sum to 1.
    """

    state_dicts = list(state_dicts)
    if not state_dicts:
        raise ValueError("No state_dicts provided for aggregation.")

    avg_state: Dict[str, Tensor] = {}
    for key in state_dicts[0][0]:
        accumulator: Tensor | None = None
        for state, weight in state_dicts:
            value = state[key] * weight
            accumulator = value if accumulator is None else accumulator + value
        if accumulator is None:
            raise RuntimeError("Failed to aggregate state dicts.")
        avg_state[key] = accumulator
    return avg_state
