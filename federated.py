"""
Federated optimization routines for the FedLoNS framework.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import _compat  # noqa: F401
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from config import DataConfig, FederatedConfig, ModelConfig, OptimizerConfig
from model import FedLoNSDiffusionHead, FedLoNSDriftModel, FedLoNSModel
from utils import MetricTracker, gaussian_nll_loss, weighted_average_state_dict


def _dataloader(
    dataset: Dataset,
    cfg: DataConfig,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )


@dataclass
class LocalTrainingResult:
    state_dict: Dict[str, Tensor]
    train_loss: float


class FederatedClient:
    """Client holding a personalized diffusion head and local datasets."""

    def __init__(
        self,
        client_id: int,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        optim_cfg: OptimizerConfig,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.device = device
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader = _dataloader(train_dataset, data_cfg, shuffle=True)
        self.val_loader = (
            _dataloader(val_dataset, data_cfg, shuffle=False) if val_dataset else None
        )
        self.diffusion = FedLoNSDiffusionHead(
            hidden_size=model_cfg.hidden_size,
            diffusion_dim=model_cfg.diffusion_dim,
        ).to(device)

    @property
    def num_train_samples(self) -> int:
        return len(self.train_dataset)

    def build_local_model(self, drift_state: Dict[str, Tensor]) -> FedLoNSModel:
        drift = FedLoNSDriftModel(self.model_cfg).to(self.device)
        drift.load_state_dict(drift_state)
        return FedLoNSModel(drift, self.diffusion)

    def local_train(
        self, drift_state: Dict[str, Tensor], fed_cfg: FederatedConfig
    ) -> LocalTrainingResult:
        model = self.build_local_model(drift_state)
        model.train()
        optimizer = torch.optim.Adam(
            [
                {"params": model.drift.parameters()},
                {"params": model.diffusion.parameters()},
            ],
            lr=self.optim_cfg.lr,
            weight_decay=self.optim_cfg.weight_decay,
            betas=self.optim_cfg.betas,
        )

        loss_meter = MetricTracker()
        for epoch in range(fed_cfg.local_epochs):
            pbar = tqdm(
                self.train_loader,
                desc=f"Client {self.client_id} Epoch {epoch + 1}/{fed_cfg.local_epochs}",
                leave=False,
                mininterval=10,
            )
            for batch in pbar:
                inputs = batch["inputs"].to(self.device)
                target_log = batch["target_log"].to(self.device)

                optimizer.zero_grad()
                mean_log, std = model(inputs)
                loss = gaussian_nll_loss(target_log, mean_log, std).mean()
                loss.backward()
                if fed_cfg.grad_clip_value:
                    clip_grad_norm_(model.parameters(), fed_cfg.grad_clip_value)
                optimizer.step()

                loss_meter.update(loss.item(), inputs.size(0))
                pbar.set_postfix(loss=f"{loss_meter.average:.4f}")

        updated_state = copy.deepcopy(model.drift.state_dict())
        return LocalTrainingResult(state_dict=updated_state, train_loss=loss_meter.average)

    def evaluate(self, drift_state: Dict[str, Tensor]) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        model = self.build_local_model(drift_state)
        model.eval()

        total_samples = 0
        mae_total = 0.0
        mse_total = 0.0
        peak_error = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["inputs"].to(self.device)
                target_log = batch["target_log"].to(self.device)
                target_mmol = batch["target_mmol"].to(self.device)

                mean_log, std = model(inputs)
                _ = gaussian_nll_loss(target_log, mean_log, std).mean()
                pred_mmol = torch.exp(mean_log)
                abs_error = torch.abs(pred_mmol - target_mmol)

                batch_size = inputs.size(0)
                total_samples += batch_size
                mae_total += abs_error.sum().item()
                mse_total += torch.sum((pred_mmol - target_mmol) ** 2).item()
                peak_error = max(peak_error, abs_error.max().item())

        if total_samples == 0:
            return {}

        mae = mae_total / total_samples
        rmse = math.sqrt(mse_total / total_samples)

        return {
            "mae": mae,
            "rmse": rmse,
            "pe": peak_error,
            "samples": total_samples,
            "mae_sum": mae_total,
            "mse_sum": mse_total,
        }


class FedLoNSServer:
    """Coordinator that aggregates shared drift parameters across clients."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        fed_cfg: FederatedConfig,
        optim_cfg: OptimizerConfig,
        clients: Sequence[FederatedClient],
        device: torch.device,
    ) -> None:
        self.model_cfg = model_cfg
        self.fed_cfg = fed_cfg
        self.optim_cfg = optim_cfg
        self.clients = list(clients)
        if not self.clients:
            raise ValueError("At least one client is required.")
        self.device = device
        self.global_drift = FedLoNSDriftModel(model_cfg).to(device)

    @property
    def drift_state(self) -> Dict[str, Tensor]:
        return copy.deepcopy(self.global_drift.state_dict())

    def select_clients(self) -> List[FederatedClient]:
        if (
            self.fed_cfg.clients_per_round is None
            or self.fed_cfg.clients_per_round >= len(self.clients)
        ):
            return list(self.clients)
        indices = random.sample(range(len(self.clients)), self.fed_cfg.clients_per_round)
        return [self.clients[idx] for idx in indices]

    def aggregate(self, weighted_states: Iterable[tuple[Dict[str, Tensor], float]]) -> None:
        aggregated = weighted_average_state_dict(weighted_states)
        self.global_drift.load_state_dict(aggregated)

    def train(self) -> List[dict[str, float]]:
        history: List[dict[str, float]] = []
        round_bar = tqdm(range(self.fed_cfg.rounds), desc="Federated rounds")
        for round_idx in round_bar:
            drift_state = self.drift_state
            selected_clients = self.select_clients()
            total_weight = sum(client.num_train_samples for client in selected_clients)

            client_states = []
            round_losses = []
            for client in selected_clients:
                result = client.local_train(drift_state, self.fed_cfg)
                weight = client.num_train_samples / max(total_weight, 1)
                client_states.append((result.state_dict, weight))
                round_losses.append(result.train_loss)

            self.aggregate(client_states)
            avg_train_loss = sum(round_losses) / max(len(round_losses), 1)
            eval_metrics = self.evaluate()
            metrics = {"round": round_idx + 1, "train_loss": avg_train_loss}
            metrics.update(eval_metrics)
            history.append(metrics)
            round_bar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items() if isinstance(v, float)})
        return history

    def evaluate(self) -> dict[str, float]:
        metrics = {}
        total_samples = 0
        mae_sum = 0.0
        mse_sum = 0.0
        peak_sum = 0.0
        drift_state = self.drift_state
        any_eval = False
        for client in self.clients:
            client_metrics = client.evaluate(drift_state)
            if not client_metrics:
                continue
            any_eval = True
            samples = client_metrics.get("samples", 0)
            if samples <= 0:
                continue
            total_samples += samples
            mae_sum += client_metrics.get("mae_sum", client_metrics["mae"] * samples)
            mse_sum += client_metrics.get("mse_sum", (client_metrics["rmse"] ** 2) * samples)
            peak_sum += client_metrics["pe"] * samples

        if any_eval and total_samples > 0:
            metrics = {
                "mae": mae_sum / total_samples,
                "rmse": math.sqrt(mse_sum / total_samples),
                "pe": peak_sum / total_samples,
            }
        return metrics
