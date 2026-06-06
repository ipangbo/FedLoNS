# FedLoNS

Federated Log-Linear Noise SDE (FedLoNS) is a PyTorch prototype for forecasting continuous glucose monitoring (CGM) trajectories under federated, heterogeneous, and irregular sampling conditions. The implementation follows the formulation described in the accompanying paper excerpt and organises the codebase into reusable modules for data simulation, model definition, and federated optimisation.

## Project Layout

- `config.py` – dataclass-based configuration objects for data, model, optimisation, and protocol settings.
- `data.py` – irregular CGM data simulation and sliding-window dataset utilities.
- `model.py` – shared LSTM drift backbone and client-specific diffusion head.
- `federated.py` – client/server training loops with weighted aggregation of drift parameters.
- `utils.py` – helpers for seeding, device resolution, Gaussian NLL, and parameter aggregation.
- `examples/demo.py` – end-to-end demo that simulates five clients, runs the federated protocol, and reports metrics.
- Evaluation reports mean absolute error (MAE), root mean squared error (RMSE), and peak-error (PE) to capture safety-critical performance.

All heavy computations rely on PyTorch with CUDA support when available and fall back to CPU automatically.

## Prerequisites

- Python 3.10+
- PyTorch with CUDA (optional) and its dependencies
- NumPy, tqdm

> **Note:** some constrained environments expose a namespace-only NumPy stub. If so, ensure a full NumPy installation is available before running the demo.

## Running the Demo

```bash
python examples/demo.py
```

The script:

1. Seeds all random number generators for reproducibility.
2. Simulates five personalised CGM clients in mmol/L with irregular sampling.
3. Trains the FedLoNS model for several federated rounds using tqdm progress bars.
4. Prints per-round aggregated metrics (training loss, MAE, RMSE, Peak-Error) and final per-client evaluation statistics.

Configuration is expressed through dataclasses, making it straightforward to customise hyper-parameters directly in the script or from user code.

## Extending

- Adjust data simulation parameters in `examples/demo.py` or call `simulate_client_cgm_series` directly to plug in real CGM logs.
- Replace the LSTM backbone or diffusion head by subclassing the components in `fedlons/model.py`.
- Integrate custom evaluation metrics by extending `FederatedClient.evaluate`.

## Results
| Method | Aleppo M | Aleppo R | Aleppo P | Brown M | Brown R | Brown P | Colas M | Colas R | Colas P | Beijing AQ M | Beijing AQ R | Beijing AQ P | London AQ M | London AQ R | London AQ P | METR-LA M | METR-LA R | METR-LA P | PEMS-BAY M | PEMS-BAY R | PEMS-BAY P |
|----------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|-------------:|-------------:|-------------:|------------:|------------:|------------:|----------:|----------:|----------:|------------:|------------:|------------:|
| **Federated Learning Methods** |||||||||||||||||||||
| Local-Only | 0.71 | 1.38 | 21.70 | 0.45 | 0.81 | 21.32 | 1.49 | 2.20 | 6.23 | 1.93 | 2.37 | 9.31 | 0.72 | 1.44 | 22.08 | 0.85 | 0.97 | 15.19 | 0.43 | 0.77 | 20.42 |
| Centralized | 0.52 | 1.09 | 20.17 | 0.66 | 1.18 | 17.11 | 2.19 | 2.97 | 5.12 | 2.19 | 2.91 | 8.29 | 0.65 | 1.24 | 16.66 | 0.64 | 1.16 | 16.53 | 1.79 | 2.26 | 15.25 |
| FedAvg | 0.42 | 0.66 | 13.16 | 0.34 | 0.64 | 13.01 | 1.15 | 1.51 | 3.01 | 0.45 | 0.74 | 12.80 | 0.85 | 1.58 | 17.50 | 0.62 | 0.98 | 13.25 | 0.32 | 0.66 | 12.88 |
| FedProx | 0.34 | 0.59 | 12.48 | 0.31 | 0.53 | 11.43 | 1.17 | 1.26 | 3.05 | 0.44 | 0.69 | 12.28 | 0.29 | 1.50 | 12.58 | 0.60 | 0.94 | 16.17 | 0.59 | 0.87 | 16.22 |
| pFedMe | 0.43 | 0.69 | 13.02 | 0.37 | 0.61 | 12.34 | 1.22 | 1.66 | 3.34 | 0.39 | 0.60 | 12.97 | 0.97 | 1.20 | 12.92 | 0.53 | 0.89 | 13.85 | 1.00 | 1.32 | 16.97 |
| FedRep | 0.41 | 0.72 | 13.86 | 0.36 | 0.67 | 13.86 | 1.26 | 1.57 | 3.00 | 0.78 | 1.40 | 8.35 | 0.50 | 0.86 | 13.29 | 0.86 | 1.38 | 14.04 | 0.49 | 0.83 | 14.72 |
| FedLoSD | **0.31** | **0.55** | **11.37** | **0.27** | **0.52** | **11.33** | **0.28** | **0.36** | **0.80** | **0.27** | **0.36** | **6.85** | **0.26** | **0.53** | **11.44** | **0.48** | **0.77** | **12.30** | **0.24** | **0.40** | **10.04** |

| Method | Aleppo M | Aleppo R | Aleppo P | Brown M | Brown R | Brown P | Colas M | Colas R | Colas P | Beijing AQ M | Beijing AQ R | Beijing AQ P | London AQ M | London AQ R | London AQ P | METR-LA M | METR-LA R | METR-LA P | PEMS-BAY M | PEMS-BAY R | PEMS-BAY P |
|----------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|-------------:|-------------:|-------------:|------------:|------------:|------------:|----------:|----------:|----------:|------------:|------------:|------------:|
| **ODE Methods (Original Data)** |||||||||||||||||||||
| GRU-D | 0.57 | 0.69 | 21.49 | 0.43 | 0.87 | 17.54 | 0.46 | 0.62 | 1.50 | 0.58 | 0.74 | 19.90 | 0.58 | 0.83 | 15.64 | 0.93 | 0.64 | 16.45 | 0.79 | 1.10 | 22.01 |
| ODE-RNN | 0.49 | 1.02 | 18.33 | 0.42 | 0.86 | 15.53 | 0.44 | 0.65 | 1.56 | 0.62 | 1.18 | 15.76 | 0.47 | 0.87 | 15.42 | 0.82 | 1.27 | 20.80 | 0.33 | 0.62 | 12.78 |
| Neural SDE | 0.52 | 0.88 | 20.34 | 0.46 | 0.86 | 17.41 | 0.41 | 0.62 | 1.07 | 0.46 | 0.84 | 17.83 | 0.59 | 0.84 | 19.68 | 0.43 | 0.91 | 15.55 | 0.67 | 1.19 | 22.50 |
| Log-SDE | 0.47 | 0.87 | 14.68 | 0.53 | 0.87 | 19.14 | 0.40 | 0.49 | 1.25 | 0.66 | 1.36 | 17.88 | 0.38 | 0.59 | 12.32 | 0.49 | 0.78 | 20.10 | 0.37 | 0.47 | 14.19 |
| LNSDE | 0.51 | 0.75 | 19.43 | 0.39 | 0.83 | 17.01 | 0.50 | 0.56 | 1.50 | 0.48 | 0.53 | 12.77 | 0.61 | 1.19 | 17.38 | 0.84 | 1.20 | 20.47 | 0.49 | 0.73 | 21.12 |
| FedLoSD | **0.31** | **0.55** | **11.37** | **0.27** | **0.52** | **11.33** | **0.28** | **0.36** | **0.80** | **0.27** | **0.36** | **6.85** | **0.26** | **0.53** | **11.44** | **0.48** | **0.77** | **12.30** | **0.24** | **0.40** | **10.04** |

| Method | Aleppo M | Aleppo R | Aleppo P | Brown M | Brown R | Brown P | Colas M | Colas R | Colas P | Beijing AQ M | Beijing AQ R | Beijing AQ P | London AQ M | London AQ R | London AQ P | METR-LA M | METR-LA R | METR-LA P | PEMS-BAY M | PEMS-BAY R | PEMS-BAY P |
|----------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|-------------:|-------------:|-------------:|------------:|------------:|------------:|----------:|----------:|----------:|------------:|------------:|------------:|
| **ODE Methods (15-min Interval)** |||||||||||||||||||||
| GRU-D | 1.00 | 1.56 | 21.84 | 0.87 | 1.31 | 25.73 | 0.77 | 1.12 | 1.97 | 0.98 | 1.36 | 13.27 | 0.97 | 1.37 | 25.18 | 0.73 | 1.13 | 15.12 | 0.96 | 1.44 | 15.20 |
| ODE-RNN | 0.79 | 1.35 | 20.47 | 0.98 | 1.19 | 21.92 | 0.85 | 0.83 | 1.69 | 0.93 | 2.02 | 13.20 | 1.27 | 2.43 | 19.34 | 0.85 | 1.88 | 14.61 | 0.55 | 1.00 | 15.91 |
| Neural SDE | 0.74 | 1.51 | 22.69 | 0.73 | 1.44 | 26.58 | 0.88 | 0.92 | 1.85 | 0.86 | 1.42 | 15.50 | 0.90 | 1.94 | 12.76 | 1.34 | 1.76 | 16.45 | 0.76 | 1.37 | 26.70 |
| FedLoSD | **0.55** | **0.99** | **18.79** | **0.51** | **0.85** | **14.25** | **0.48** | **0.61** | **1.15** | **0.47** | **0.59** | **11.09** | **0.56** | **1.00** | **10.57** | **0.62** | **1.10** | **11.16** | **0.53** | **0.92** | **14.40** |

| Method | Aleppo M | Aleppo R | Aleppo P | Brown M | Brown R | Brown P | Colas M | Colas R | Colas P | Beijing AQ M | Beijing AQ R | Beijing AQ P | London AQ M | London AQ R | London AQ P | METR-LA M | METR-LA R | METR-LA P | PEMS-BAY M | PEMS-BAY R | PEMS-BAY P |
|----------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|-------------:|-------------:|-------------:|------------:|------------:|------------:|----------:|----------:|----------:|------------:|------------:|------------:|
| **ODE Methods (30-min Interval)** |||||||||||||||||||||
| GRU-D | 0.96 | 1.94 | 22.18 | 0.90 | 1.33 | 23.76 | 0.75 | 1.21 | 1.69 | 0.93 | 1.27 | 23.87 | 1.18 | 1.74 | 23.48 | 0.83 | 1.33 | 15.82 | 0.73 | 1.35 | 19.68 |
| ODE-RNN | 0.82 | 1.66 | 21.39 | 1.02 | 1.26 | 21.20 | 0.74 | 1.07 | 1.89 | 0.89 | 1.64 | 17.47 | 0.73 | 1.08 | 12.48 | 0.65 | 0.89 | 17.29 | 0.84 | 1.37 | 20.05 |
| Neural SDE | 0.78 | 1.43 | 23.49 | 0.83 | 1.28 | 27.25 | 0.84 | 1.12 | 1.95 | 0.98 | 1.52 | 16.69 | 1.13 | 1.54 | 15.47 | 0.68 | 1.14 | 17.25 | 0.80 | 1.07 | 15.15 |
| FedLoSD | **0.65** | **1.22** | **15.85** | **0.80** | **1.13** | **15.72** | **0.64** | **0.83** | **1.36** | **0.79** | **0.98** | **15.39** | **0.66** | **0.86** | **11.29** | **0.42** | **0.56** | **13.83** | **0.74** | **1.03** | **14.09** |
