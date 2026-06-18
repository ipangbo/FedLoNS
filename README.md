# FedLoSD

Federated Log-State Dynamics (FedLoSD) is a PyTorch prototype for federated forecasting over heterogeneous and irregularly sampled sensor time series. The repository supports experiments on continuous glucose monitoring (CGM), air-quality monitoring, and traffic sensor-network forecasting tasks.

The implementation follows the formulation described in the accompanying paper and organises the codebase into reusable modules for data simulation/preprocessing, model definition, federated optimisation, and evaluation.

## Project Layout

* `config.py` – dataclass-based configuration objects for data, model, optimisation, and protocol settings.
* `data.py` – irregular sensor time-series simulation, preprocessing, and sliding-window dataset utilities.
* `model.py` – shared recurrent drift backbone and client-specific diffusion/noise head.
* `federated.py` – client/server training loops with weighted aggregation of shared drift parameters.
* `utils.py` – helpers for seeding, device resolution, Gaussian NLL, metrics, and parameter aggregation.
* `examples/demo.py` – an end-to-end demo that simulates heterogeneous irregular clients, runs the federated protocol, and reports evaluation metrics.

All heavy computations rely on PyTorch with CUDA support when available and fall back to CPU automatically.

## Prerequisites

* Python 3.10+
* PyTorch with CUDA support, optional but recommended
* NumPy
* tqdm

> **Note:** Some constrained environments expose a namespace-only NumPy stub. If so, ensure that a full NumPy installation is available before running the demo.

## Installation

```bash
git clone <REPOSITORY_URL>
cd FedLoSD

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

If `requirements.txt` is not available in your local copy, install the core dependencies manually:

```bash
pip install torch numpy tqdm
```

## Running the Demo

```bash
python examples/demo.py
```

The demo script illustrates the main federated training pipeline on simulated irregular sensor clients. The full paper results are summarised below. Dataset-specific preprocessing and experiment configuration should be adjusted according to the dataset used.

Configuration is expressed through dataclasses, making it straightforward to customise hyperparameters directly in the script or from user code.

## Datasets

The paper evaluates FedLoSD on seven sensor datasets: three CGM datasets, two air-quality datasets, and two traffic sensor-network datasets.

| Dataset | Domain | Source / Link | Notes |
|---|---|---|---|
| Aleppo | CGM | https://doi.org/10.2337/dc16-2482 | CGM data/reference associated with the REPLACE-BG randomized trial in adults with well-controlled type 1 diabetes. |
| Brown | CGM | https://www.nejm.org/doi/full/10.1056/NEJMoa1907863 | CGM data/reference associated with the six-month randomized multicenter closed-loop control trial in type 1 diabetes. |
| Colas | CGM | https://github.com/IrinaStatsLab/GlucoBench | Public CGM benchmark resource listing Colas and related CGM datasets/preprocessing resources. |
| Beijing AQ | Air quality | https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data | Beijing Multi-Site Air Quality dataset from UCI. |
| London AQ | Air quality | https://www.londonair.org.uk/london/asp/datadownload.asp | LondonAir / London Air Quality Network data download page. |
| METR-LA | Traffic sensor network | https://github.com/liyaguang/DCRNN | The DCRNN repository provides instructions and links for `metr-la.h5`. |
| PEMS-BAY | Traffic sensor network | https://github.com/liyaguang/DCRNN | The DCRNN repository provides instructions and links for `pems-bay.h5`. |
For reproducibility, place processed datasets under a consistent local directory, for example:

```text
data/
  cgm/
    aleppo/
    brown/
    colas/
  air_quality/
    beijing/
    london/
  traffic/
    METR-LA/
    PEMS-BAY/
```

## Evaluation Metrics

Evaluation reports:

* Mean Absolute Error (MAE, denoted as `M`)
* Root Mean Squared Error (RMSE, denoted as `R`)
* Peak Error (PE, denoted as `P`)

Peak Error is included to capture safety-critical worst-case deviations, which are particularly important for sensor forecasting applications such as CGM and traffic monitoring.

All reported results are averaged over three random seeds.

## Results

### Federated Learning Methods

| Method      | Aleppo M | Aleppo R |  Aleppo P |  Brown M |  Brown R |   Brown P |  Colas M |  Colas R |  Colas P | Beijing AQ M | Beijing AQ R | Beijing AQ P | London AQ M | London AQ R | London AQ P | METR-LA M | METR-LA R | METR-LA P | PEMS-BAY M | PEMS-BAY R | PEMS-BAY P |
| ----------- | -------: | -------: | --------: | -------: | -------: | --------: | -------: | -------: | -------: | -----------: | -----------: | -----------: | ----------: | ----------: | ----------: | --------: | --------: | --------: | ---------: | ---------: | ---------: |
| Local-Only  |     0.71 |     1.38 |     21.70 |     0.45 |     0.81 |     21.32 |     1.49 |     2.20 |     6.23 |         1.93 |         2.37 |         9.31 |        0.72 |        1.44 |       22.08 |      0.85 |      0.97 |     15.19 |       0.43 |       0.77 |      20.42 |
| Centralized |     0.52 |     1.09 |     20.17 |     0.66 |     1.18 |     17.11 |     2.19 |     2.97 |     5.12 |         2.19 |         2.91 |         8.29 |        0.65 |        1.24 |       16.66 |      0.64 |      1.16 |     16.53 |       1.79 |       2.26 |      15.25 |
| FedAvg      |     0.42 |     0.66 |     13.16 |     0.34 |     0.64 |     13.01 |     1.15 |     1.51 |     3.01 |         0.45 |         0.74 |        12.80 |        0.85 |        1.58 |       17.50 |      0.62 |      0.98 |     13.25 |       0.32 |       0.66 |      12.88 |
| FedProx     |     0.34 |     0.59 |     12.48 |     0.31 |     0.53 |     11.43 |     1.17 |     1.26 |     3.05 |         0.44 |         0.69 |        12.28 |        0.29 |        1.50 |       12.58 |      0.60 |      0.94 |     16.17 |       0.59 |       0.87 |      16.22 |
| pFedMe      |     0.43 |     0.69 |     13.02 |     0.37 |     0.61 |     12.34 |     1.22 |     1.66 |     3.34 |         0.39 |         0.60 |        12.97 |        0.97 |        1.20 |       12.92 |      0.53 |      0.89 |     13.85 |       1.00 |       1.32 |      16.97 |
| FedRep      |     0.41 |     0.72 |     13.86 |     0.36 |     0.67 |     13.86 |     1.26 |     1.57 |     3.00 |         0.78 |         1.40 |         8.35 |        0.50 |        0.86 |       13.29 |      0.86 |      1.38 |     14.04 |       0.49 |       0.83 |      14.72 |
| **FedLoSD** | **0.31** | **0.55** | **11.37** | **0.27** | **0.52** | **11.33** | **0.28** | **0.36** | **0.80** |     **0.27** |     **0.36** |     **6.85** |    **0.26** |    **0.53** |   **11.44** |  **0.48** |  **0.77** | **12.30** |   **0.24** |   **0.40** |  **10.04** |

### ODE Methods on Original Irregular Data

| Method      | Aleppo M | Aleppo R |  Aleppo P |  Brown M |  Brown R |   Brown P |  Colas M |  Colas R |  Colas P | Beijing AQ M | Beijing AQ R | Beijing AQ P | London AQ M | London AQ R | London AQ P | METR-LA M | METR-LA R | METR-LA P | PEMS-BAY M | PEMS-BAY R | PEMS-BAY P |
| ----------- | -------: | -------: | --------: | -------: | -------: | --------: | -------: | -------: | -------: | -----------: | -----------: | -----------: | ----------: | ----------: | ----------: | --------: | --------: | --------: | ---------: | ---------: | ---------: |
| GRU-D       |     0.57 |     0.69 |     21.49 |     0.43 |     0.87 |     17.54 |     0.46 |     0.62 |     1.50 |         0.58 |         0.74 |        19.90 |        0.58 |        0.83 |       15.64 |      0.93 |      1.64 |     16.45 |       0.79 |       1.10 |      22.01 |
| ODE-RNN     |     0.49 |     1.02 |     18.33 |     0.42 |     0.86 |     15.53 |     0.44 |     0.65 |     1.56 |         0.62 |         1.18 |        15.76 |        0.47 |        0.87 |       15.42 |      0.82 |      1.27 |     20.80 |       0.33 |       0.62 |      12.78 |
| Neural SDE  |     0.52 |     0.88 |     20.34 |     0.46 |     0.86 |     17.41 |     0.41 |     0.62 |     1.07 |         0.46 |         0.84 |        17.83 |        0.59 |        0.84 |       19.68 |      0.43 |      0.91 |     15.55 |       0.67 |       1.19 |      22.50 |
| Log-SDE     |     0.47 |     0.87 |     14.68 |     0.53 |     0.87 |     19.14 |     0.40 |     0.49 |     1.25 |         0.66 |         1.36 |        17.88 |        0.38 |        0.59 |       12.32 |      0.49 |      0.78 |     20.10 |       0.37 |       0.47 |      14.19 |
| LNSDE       |     0.51 |     0.75 |     19.43 |     0.39 |     0.83 |     17.01 |     0.50 |     0.56 |     1.50 |         0.48 |         0.53 |        12.77 |        0.61 |        1.19 |       17.38 |      0.84 |      1.20 |     20.47 |       0.49 |       0.73 |      21.12 |
| **FedLoSD** | **0.31** | **0.55** | **11.37** | **0.27** | **0.52** | **11.33** | **0.28** | **0.36** | **0.80** |     **0.27** |     **0.36** |     **6.85** |    **0.26** |    **0.53** |   **11.44** |  **0.48** |  **0.77** | **12.30** |   **0.24** |   **0.40** |  **10.04** |

### ODE Methods on 15-Minute Intervals

| Method      | Aleppo M | Aleppo R |  Aleppo P |  Brown M |  Brown R |   Brown P |  Colas M |  Colas R |  Colas P | Beijing AQ M | Beijing AQ R | Beijing AQ P | London AQ M | London AQ R | London AQ P | METR-LA M | METR-LA R | METR-LA P | PEMS-BAY M | PEMS-BAY R | PEMS-BAY P |
| ----------- | -------: | -------: | --------: | -------: | -------: | --------: | -------: | -------: | -------: | -----------: | -----------: | -----------: | ----------: | ----------: | ----------: | --------: | --------: | --------: | ---------: | ---------: | ---------: |
| GRU-D       |     1.00 |     1.56 |     21.84 |     0.87 |     1.31 |     25.73 |     0.77 |     1.12 |     1.97 |         0.98 |         1.36 |        13.27 |        0.97 |        1.37 |       25.18 |      0.73 |      1.13 |     15.12 |       0.96 |       1.44 |      15.20 |
| ODE-RNN     |     0.79 |     1.35 |     20.47 |     0.98 |     1.19 |     21.92 |     0.85 |     1.26 |     1.69 |         0.93 |         2.02 |        13.20 |        1.27 |        2.43 |       19.34 |      0.85 |      1.88 |     14.61 |       0.55 |       1.00 |      15.91 |
| Neural SDE  |     0.74 |     1.51 |     22.69 |     0.73 |     1.44 |     26.58 |     0.88 |     0.92 |     1.85 |         0.86 |         1.42 |        15.50 |        0.90 |        1.94 |       12.76 |      1.34 |      1.76 |     16.45 |       0.76 |       1.37 |      26.70 |
| **FedLoSD** | **0.55** | **0.99** | **18.79** | **0.51** | **0.85** | **14.25** | **0.48** | **0.61** | **1.15** |     **0.47** |     **0.59** |    **11.09** |    **0.56** |    **1.00** |   **10.57** |  **0.62** |  **1.10** | **11.16** |   **0.53** |   **0.92** |  **14.40** |

### ODE Methods on 30-Minute Intervals

| Method      | Aleppo M | Aleppo R |  Aleppo P |  Brown M |  Brown R |   Brown P |  Colas M |  Colas R |  Colas P | Beijing AQ M | Beijing AQ R | Beijing AQ P | London AQ M | London AQ R | London AQ P | METR-LA M | METR-LA R | METR-LA P | PEMS-BAY M | PEMS-BAY R | PEMS-BAY P |
| ----------- | -------: | -------: | --------: | -------: | -------: | --------: | -------: | -------: | -------: | -----------: | -----------: | -----------: | ----------: | ----------: | ----------: | --------: | --------: | --------: | ---------: | ---------: | ---------: |
| GRU-D       |     0.96 |     1.94 |     22.18 |     0.90 |     1.33 |     23.76 |     0.75 |     1.21 |     1.69 |         0.93 |         1.27 |        23.87 |        1.18 |        1.74 |       23.48 |      0.83 |      1.33 |     15.82 |       0.73 |       1.35 |      19.68 |
| ODE-RNN     |     0.82 |     1.66 |     21.39 |     1.02 |     1.26 |     21.20 |     0.74 |     1.07 |     1.89 |         0.89 |         1.64 |        17.47 |        0.73 |        1.08 |       12.48 |      0.65 |      0.89 |     17.29 |       0.84 |       1.37 |      20.05 |
| Neural SDE  |     0.78 |     1.43 |     23.49 |     0.83 |     1.28 |     27.25 |     0.84 |     1.12 |     1.95 |         0.98 |         1.52 |        16.69 |        1.13 |        1.54 |       15.47 |      0.68 |      1.14 |     17.25 |       0.80 |       1.07 |      15.15 |
| **FedLoSD** | **0.65** | **1.22** | **15.85** | **0.80** | **1.13** | **15.72** | **0.64** | **0.83** | **1.36** |     **0.79** |     **0.98** |    **15.39** |    **0.66** |    **0.86** |   **11.29** |  **0.42** |  **0.56** | **13.83** |   **0.74** |   **1.03** |  **14.09** |

## Notes on Uniform-Interval Evaluation

The diffusion structure of Log-SDE and LNSDE is strongly coupled with the original irregular observation intervals. Applying these models directly to fixed 15-minute or 30-minute sampling can lead to unstable or unfair dynamics. Therefore, Log-SDE and LNSDE are excluded from the uniform-interval comparison blocks.

FedLoSD is reported in both the federated-learning comparison and the continuous-time/original-data comparison because it is the proposed federated method and also the proposed irregular-time dynamics model.

## Extending the Codebase

* Adjust data simulation or preprocessing parameters in `examples/demo.py` or `data.py`.
* Plug in real sensor logs by implementing dataset-specific loaders that return timestamped multivariate series.
* Replace the recurrent drift backbone or diffusion/noise head by subclassing the components in `model.py`.
* Add new federated optimisation strategies in `federated.py`.
* Integrate custom evaluation metrics by extending `FederatedClient.evaluate`.

