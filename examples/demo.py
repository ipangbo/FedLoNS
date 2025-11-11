"""
Demo script that simulates client CGM data, runs FedLoNS training, and reports metrics.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List

import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if not hasattr(np, "ndarray"):
    class _NumPyArrayPlaceholder:
        pass

    np.ndarray = _NumPyArrayPlaceholder  # type: ignore[attr-defined]

from fedlons import (
    CGMDataset,
    DataConfig,
    FedLoNSServer,
    FederatedClient,
    FederatedConfig,
    ModelConfig,
    OptimizerConfig,
    SeedConfig,
    get_torch_device,
    set_random_seed,
    simulate_client_cgm_series,
)

import torch


def build_clients(
    num_clients: int,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    optim_cfg: OptimizerConfig,
    device: torch.device,
) -> List[FederatedClient]:
    clients: List[FederatedClient] = []
    rng = np.random.default_rng()

    for client_id in range(num_clients):
        base_glucose = float(np.clip(rng.normal(loc=6.0, scale=0.8), 4.5, 9.0))
        circadian = float(np.clip(rng.normal(loc=0.3, scale=0.1), 0.1, 0.6))
        drift_strength = float(np.clip(rng.normal(loc=0.015, scale=0.004), 0.008, 0.03))
        diffusion_scale = float(np.clip(rng.normal(loc=0.08, scale=0.03), 0.04, 0.14))
        irregularity = float(np.clip(rng.normal(loc=5.0, scale=1.2), 2.0, 8.0))

        times, glucose = simulate_client_cgm_series(
            num_points=720,
            base_glucose_mmol=base_glucose,
            circadian_amplitude=circadian,
            drift_strength=drift_strength,
            diffusion_scale=diffusion_scale,
            irregularity_scale=irregularity,
            measurement_noise=0.08,
            seed=client_id * 17 + 13,
        )

        dataset = CGMDataset(times, glucose, data_cfg)
        train_len = max(1, int(len(dataset) * data_cfg.train_fraction))
        val_len = len(dataset) - train_len
        if val_len == 0:
            val_len = 1
            train_len = len(dataset) - 1
        generator = torch.Generator().manual_seed(client_id * 97 + 7)
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_len, val_len], generator=generator
        )

        client = FederatedClient(
            client_id=client_id,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            device=device,
        )
        clients.append(client)

    return clients


def main() -> None:
    seed_cfg = SeedConfig(seed=42)
    set_random_seed(seed_cfg.seed)
    device = get_torch_device()

    data_cfg = DataConfig(
        context_length=12,
        prediction_horizon=1,
        train_fraction=0.8,
        batch_size=128,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model_cfg = ModelConfig(
        input_dim=2,
        hidden_size=96,
        num_layers=1,
        dropout=0.1,
        output_dim=1,
        diffusion_dim=1,
    )
    optim_cfg = OptimizerConfig(lr=3e-3, weight_decay=1e-4)
    fed_cfg = FederatedConfig(rounds=12, local_epochs=5, clients_per_round=None, grad_clip_value=5.0)

    print("Running FedLoNS demo with configuration:")
    for name, cfg in [
        ("SeedConfig", seed_cfg),
        ("DataConfig", data_cfg),
        ("ModelConfig", model_cfg),
        ("OptimizerConfig", optim_cfg),
        ("FederatedConfig", fed_cfg),
    ]:
        print(f"  {name}: {asdict(cfg)}")
    print(f"  Device: {device}")

    clients = build_clients(
        num_clients=5,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        device=device,
    )

    server = FedLoNSServer(
        model_cfg=model_cfg,
        fed_cfg=fed_cfg,
        optim_cfg=optim_cfg,
        clients=clients,
        device=device,
    )

    history = server.train()

    print("\nFederated training complete. Round metrics:")
    for entry in history:
        metrics_order = ["train_loss", "mae", "rmse", "pe"]
        metrics_str = ", ".join(
            f"{key}: {entry[key]:.4f}" for key in metrics_order if key in entry
        )
        print(f"  Round {entry['round']:02d} -> {metrics_str}")

    print("\nPer-client evaluation:")
    drift_state = server.drift_state
    for client in clients:
        metrics = client.evaluate(drift_state)
        if not metrics:
            continue
        client_metrics = {
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "pe": metrics["pe"],
        }
        metrics_str = ", ".join(f"{key}: {value:.4f}" for key, value in client_metrics.items())
        print(f"  Client {client.client_id}: {metrics_str}")


if __name__ == "__main__":
    main()
