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
