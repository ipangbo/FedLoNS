"""
Data simulation utilities and dataset definitions for FedLoNS.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import _compat  # noqa: F401
import numpy as np
import torch
from torch.utils.data import Dataset

from config import DataConfig


def simulate_client_cgm_series(
    num_points: int = 288,
    base_glucose_mmol: float = 5.8,
    circadian_amplitude: float = 0.3,
    drift_strength: float = 0.015,
    diffusion_scale: float = 0.08,
    irregularity_scale: float = 5.0,
    measurement_noise: float = 0.05,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate an irregularly sampled CGM trajectory using a log-linear SDE.

    Args:
        num_points: Number of observations.
        base_glucose_mmol: Baseline glucose level in mmol/L.
        circadian_amplitude: Amplitude of slow oscillations mimicking circadian rhythm.
        drift_strength: Strength of mean-reverting drift in log space.
        diffusion_scale: Multiplicative diffusion scaling.
        irregularity_scale: Mean of exponential distribution controlling time gaps (minutes).
        measurement_noise: Standard deviation of additive measurement noise (mmol/L).
        seed: Optional random seed for the simulation.
    """

    rng = np.random.default_rng(seed)
    times = np.zeros(num_points, dtype=np.float32)
    glucose = np.zeros(num_points, dtype=np.float32)

    log_base = math.log(max(base_glucose_mmol, 1e-3))
    log_value = log_base
    glucose[0] = float(math.exp(log_value))

    for idx in range(1, num_points):
        dt_minutes = rng.exponential(irregularity_scale)
        times[idx] = times[idx - 1] + dt_minutes

        circadian = circadian_amplitude * math.sin(2 * math.pi * times[idx] / (24 * 60))
        drift = drift_strength * (log_base + circadian - log_value)
        std = diffusion_scale * (1.0 + abs(log_value))
        noise = std * rng.normal() * math.sqrt(max(dt_minutes, 1e-6))
        log_value = log_value + drift * dt_minutes + noise

        mmol = float(math.exp(log_value))
        mmol += rng.normal(scale=measurement_noise)
        mmol = float(np.clip(mmol, 2.0, 18.0))  # Physiological bounds in mmol/L
        glucose[idx] = mmol

    return times, glucose


class CGMDataset(Dataset):
    """
    Sliding-window dataset over irregular CGM sequences.

    Each item provides a context window with log-glucose values and inter-sample
    time deltas, targeting the future log-glucose value after a configurable horizon.
    """

    def __init__(
        self,
        times: Iterable[float],
        glucose_mmol: Iterable[float],
        cfg: DataConfig,
    ) -> None:
        self.cfg = cfg
        self.times = np.asarray(times, dtype=np.float32)
        self.glucose = np.asarray(glucose_mmol, dtype=np.float32)

        if self.times.ndim != 1 or self.glucose.ndim != 1:
            raise ValueError("times and glucose arrays must be one-dimensional.")
        if self.times.shape[0] != self.glucose.shape[0]:
            raise ValueError("times and glucose arrays must have equal length.")

        self.log_glucose = np.log(np.clip(self.glucose, 1e-3, None)).astype(np.float32)
        deltas = np.diff(self.times, prepend=self.times[0]).astype(np.float32)
        self.deltas = deltas

        context = cfg.context_length
        horizon = cfg.prediction_horizon
        total_points = len(self.times)
        max_index = total_points - horizon
        if max_index <= context:
            raise ValueError(
                "Insufficient data points for the requested context and horizon."
            )

        self.inputs = []
        self.targets_log = []
        self.targets_mmol = []
        self.horizon_delta = []
        future_indices: List[int] = []

        for idx in range(context, max_index):
            start = idx - context
            end = idx
            future_idx = idx + horizon

            context_log = self.log_glucose[start:end]
            context_delta = self.deltas[start:end]
            features = np.stack([context_log, context_delta], axis=-1)

            self.inputs.append(features.astype(np.float32))
            self.targets_log.append(self.log_glucose[future_idx].astype(np.float32))
            self.targets_mmol.append(self.glucose[future_idx].astype(np.float32))
            future_delta = self.times[future_idx] - self.times[end - 1]
            self.horizon_delta.append(float(future_delta))
            future_indices.append(future_idx)

        self.inputs = np.stack(self.inputs)
        self.targets_log = np.asarray(self.targets_log, dtype=np.float32)
        self.targets_mmol = np.asarray(self.targets_mmol, dtype=np.float32)
        self.horizon_delta = np.asarray(self.horizon_delta, dtype=np.float32)
        self.future_indices = np.asarray(future_indices, dtype=np.int64)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": torch.from_numpy(self.inputs[index]),
            "target_log": torch.tensor(self.targets_log[index]),
            "target_mmol": torch.tensor(self.targets_mmol[index]),
            "horizon_delta": torch.tensor(self.horizon_delta[index]),
        }
