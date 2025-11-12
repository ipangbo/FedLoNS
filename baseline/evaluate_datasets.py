"""
Evaluate the FedAvg baseline on every CSV dataset found in the project's data directory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Tuple
import math

import numpy as np
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.fedavg import FedAvgClient, FedAvgServer
from config import (
    DataConfig,
    DatasetEvalConfig,
    EvaluationConfig,
    FederatedConfig,
    ModelConfig,
    OptimizerConfig,
    SeedConfig,
)
from data import CGMDataset
from utils import get_torch_device, set_random_seed

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_series(csv_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    import csv
    from datetime import datetime
    from collections import defaultdict

    records: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            glucose_str = (row.get("gl") or "").strip()
            time_str = (row.get("time") or "").strip()
            patient_id = (row.get("id") or "").strip()
            if not glucose_str or not time_str or not patient_id:
                continue
            try:
                glucose = float(glucose_str)
                timestamp = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            records[patient_id].append((timestamp, glucose))

    series: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {}
    for patient_id, entries in records.items():
        if not entries:
            continue
        entries.sort(key=lambda item: item[0])
        base_time = entries[0][0]
        times = np.array(
            [(entry[0] - base_time).total_seconds() / 60.0 for entry in entries],
            dtype=np.float32,
        )
        glucose = np.array([entry[1] for entry in entries], dtype=np.float32)
        timestamps = [entry[0].isoformat() for entry in entries]
        series[patient_id] = (times, glucose, timestamps)
    return series


def make_segment_dataset(
    times: np.ndarray,
    glucose: np.ndarray,
    start: int,
    end: int,
    cfg: DataConfig,
) -> CGMDataset:
    segment_times = times[start:end] - times[start]
    segment_glucose = glucose[start:end]
    return CGMDataset(segment_times, segment_glucose, cfg)


def prepare_clients(
    series: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    optim_cfg: OptimizerConfig,
    device: torch.device,
) -> Tuple[List[FedAvgClient], Dict[str, CGMDataset]]:
    clients: List[FedAvgClient] = []
    test_datasets: Dict[str, CGMDataset] = {}
    min_points = data_cfg.context_length + data_cfg.prediction_horizon + 1

    for patient_id, (times, glucose, _) in series.items():
        num_points = len(times)
        if num_points < 3 * min_points:
            continue

        train_len = max(min_points, int(num_points * data_cfg.train_fraction))
        max_train_len = num_points - 2 * min_points
        train_len = min(train_len, max_train_len)
        if train_len < min_points:
            continue

        remaining = num_points - train_len
        if remaining < 2 * min_points:
            continue

        val_len = max(min_points, remaining // 2)
        val_len = min(val_len, remaining - min_points)
        if val_len < min_points:
            continue

        test_len = remaining - val_len
        if test_len < min_points:
            continue

        train_end = train_len
        val_end = train_len + val_len

        try:
            train_dataset = make_segment_dataset(times, glucose, 0, train_end, data_cfg)
            val_dataset = make_segment_dataset(times, glucose, train_end, val_end, data_cfg)
            test_dataset = make_segment_dataset(times, glucose, val_end, num_points, data_cfg)
        except ValueError:
            continue

        client = FedAvgClient(
            client_id=int(patient_id),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            device=device,
        )
        clients.append(client)
        test_datasets[str(client.client_id)] = test_dataset

    return clients, test_datasets


def evaluate_dataset(
    dataset_cfg: DatasetEvalConfig,
    eval_cfg: EvaluationConfig,
    device: torch.device,
) -> Dict[str, object]:
    csv_path = (PROJECT_ROOT / dataset_cfg.csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    series = load_series(csv_path)

    data_cfg = eval_cfg.data_cfg
    if dataset_cfg.train_fraction is not None:
        data_cfg = replace(data_cfg, train_fraction=dataset_cfg.train_fraction)

    model_cfg = eval_cfg.model_cfg
    optim_cfg = eval_cfg.optim_cfg
    fed_cfg = eval_cfg.fed_cfg
    if dataset_cfg.rounds is not None or dataset_cfg.local_epochs is not None:
        fed_cfg = replace(
            fed_cfg,
            rounds=dataset_cfg.rounds or fed_cfg.rounds,
            local_epochs=dataset_cfg.local_epochs or fed_cfg.local_epochs,
        )

    clients, test_datasets = prepare_clients(
        series, data_cfg, model_cfg, optim_cfg, device
    )
    if not clients:
        raise RuntimeError(f"No clients with sufficient data in dataset '{dataset_cfg.name}'.")

    server = FedAvgServer(
        model_cfg=model_cfg,
        fed_cfg=fed_cfg,
        optim_cfg=optim_cfg,
        clients=clients,
        device=device,
    )

    history = server.train()
    model_state = server.model_state

    aggregate_samples = 0
    aggregate_mae = 0.0
    aggregate_mse = 0.0
    aggregate_pe = 0.0
    per_client_results = []

    for client in clients:
        test_dataset = test_datasets.get(str(client.client_id))
        if test_dataset is None:
            continue
        metrics = client.evaluate(model_state)
        if not metrics:
            continue
        samples = metrics["samples"]
        aggregate_samples += samples
        aggregate_mae += metrics["mae"] * samples
        aggregate_mse += (metrics["rmse"] ** 2) * samples
        aggregate_pe += metrics["pe"] * samples
        per_client_results.append(
            {
                "client_id": client.client_id,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "pe": metrics["pe"],
                "samples": samples,
            }
        )

    aggregate_result = {}
    if aggregate_samples > 0:
        aggregate_result = {
            "mae": aggregate_mae / aggregate_samples,
            "rmse": math.sqrt(aggregate_mse / aggregate_samples),
            "pe": aggregate_pe / aggregate_samples,
        }

    return {
        "dataset": dataset_cfg.name,
        "train_history": history,
        "per_client": per_client_results,
        "aggregate": aggregate_result,
        "num_clients": len(clients),
        "config": {
            "data_cfg": asdict(data_cfg),
            "model_cfg": asdict(model_cfg),
            "optim_cfg": asdict(optim_cfg),
            "fed_cfg": asdict(fed_cfg),
        },
    }


def dump_json(data: object, path: Path) -> None:
    with path.open("w") as fh:
        json.dump(data, fh, indent=2)


def default_eval_config() -> EvaluationConfig:
    data_cfg = DataConfig(
        context_length=12,
        prediction_horizon=1,
        train_fraction=0.6,
        batch_size=128,
        num_workers=0,
        pin_memory=False,
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
    fed_cfg = FederatedConfig(rounds=5, local_epochs=2, clients_per_round=40, grad_clip_value=5.0)

    dataset_configs: List[DatasetEvalConfig] = []
    for csv_path in sorted((PROJECT_ROOT / "data").glob("*.csv")):
        dataset_configs.append(
            DatasetEvalConfig(
                name=csv_path.stem,
                csv_path=str(csv_path.relative_to(PROJECT_ROOT)),
                train_fraction=0.6,
                output_filename=f"{csv_path.stem}_fedavg.json",
            )
        )

    return EvaluationConfig(
        seed_cfg=SeedConfig(seed=42),
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        fed_cfg=fed_cfg,
        datasets=tuple(dataset_configs),
    )


def main() -> None:
    eval_cfg = default_eval_config()
    set_random_seed(eval_cfg.seed_cfg.seed)
    device = get_torch_device(preferred="cuda")

    print("FedAvg evaluation configuration:")
    for name, cfg in [
        ("SeedConfig", eval_cfg.seed_cfg),
        ("DataConfig", eval_cfg.data_cfg),
        ("ModelConfig", eval_cfg.model_cfg),
        ("OptimizerConfig", eval_cfg.optim_cfg),
        ("FederatedConfig", eval_cfg.fed_cfg),
    ]:
        print(f"  {name}: {asdict(cfg)}")
    print(f"  Device: {device}")

    if not eval_cfg.datasets:
        raise RuntimeError("No datasets discovered under data/.")

    overview_entries = []
    for dataset_cfg in eval_cfg.datasets:
        print(f"\nEvaluating dataset: {dataset_cfg.name}")
        result = evaluate_dataset(dataset_cfg, eval_cfg, device)
        output_file = dataset_cfg.output_filename or f"{dataset_cfg.name}_fedavg.json"
        output_path = RESULTS_DIR / output_file
        dump_json(result, output_path)
        print(f"  Saved results to {output_path}")
        overview_entries.append(
            {
                "name": dataset_cfg.name,
                "num_clients": result["num_clients"],
                "aggregate": result["aggregate"],
                "result_file": output_file,
            }
        )

    overview_path = RESULTS_DIR / "fedavg_overview.json"
    dump_json({"datasets": overview_entries}, overview_path)
    print(f"\nSaved FedAvg overview to {overview_path}")


if __name__ == "__main__":
    main()

