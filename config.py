"""
Typed configuration objects for the FedLoNS project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SeedConfig:
    """Configuration for reproducibility."""

    seed: int = 42


@dataclass
class OptimizerConfig:
    """Optimizer hyper-parameters used by both local and global updates."""

    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    input_dim: int = 2
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    output_dim: int = 1  # Predict log-glucose mean
    diffusion_dim: int = 1  # Predict log standard deviation


@dataclass
class DataConfig:
    """Dataset parameters shared across clients."""

    context_length: int = 12
    prediction_horizon: int = 1
    train_fraction: float = 0.8
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class FederatedConfig:
    """Federated learning protocol configuration."""

    rounds: int = 10
    local_epochs: int = 1
    clients_per_round: Optional[int] = None  # Defaults to all clients
    grad_clip_value: Optional[float] = 5.0


@dataclass
class DatasetEvalConfig:
    """Configuration describing one dataset evaluation job."""

    name: str
    csv_path: str
    train_fraction: float = 0.6
    rounds: Optional[int] = None
    local_epochs: Optional[int] = None
    output_filename: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Top-level configuration for evaluating one or more datasets."""

    seed_cfg: SeedConfig = field(default_factory=SeedConfig)
    data_cfg: DataConfig = field(default_factory=DataConfig)
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    optim_cfg: OptimizerConfig = field(default_factory=OptimizerConfig)
    fed_cfg: FederatedConfig = field(default_factory=FederatedConfig)
    datasets: Tuple[DatasetEvalConfig, ...] = field(default_factory=tuple)
