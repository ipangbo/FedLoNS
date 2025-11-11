"""
Model definitions for FedLoNS including shared drift and personalized diffusion components.
"""

from __future__ import annotations

import _compat  # noqa: F401

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config import ModelConfig


class FedLoNSDriftModel(nn.Module):
    """Shared drift backbone implemented with an LSTM encoder."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.head = nn.Linear(cfg.hidden_size, cfg.output_dim)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            inputs: Tensor of shape (batch, seq_len, input_dim).

        Returns:
            mean_log: Predicted log glucose mean (batch,).
            hidden: Last hidden state representation (batch, hidden_size).
        """

        output, _ = self.lstm(inputs)
        hidden = output[:, -1, :]
        hidden = self.dropout(hidden)
        mean_log = self.head(hidden).squeeze(-1)
        return mean_log, hidden


class FedLoNSDiffusionHead(nn.Module):
    """Client-specific diffusion head producing log standard deviation."""

    def __init__(self, hidden_size: int, diffusion_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, diffusion_dim),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        log_std = self.net(hidden).squeeze(-1)
        return log_std


class FedLoNSModel(nn.Module):
    """Wrapper combining drift and diffusion modules."""

    def __init__(self, drift: FedLoNSDriftModel, diffusion: FedLoNSDiffusionHead) -> None:
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        mean_log, hidden = self.drift(inputs)
        log_std = self.diffusion(hidden)
        std = F.softplus(log_std) + 1e-3  # Ensure positivity
        return mean_log, std

