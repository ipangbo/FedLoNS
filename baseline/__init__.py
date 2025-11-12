"""
Baseline implementations that share the FedLoNS architecture but modify the
federated optimisation behaviour (e.g., FedAvg with global diffusion).
"""

from .fedavg import FedAvgClient, FedAvgServer

__all__ = ["FedAvgClient", "FedAvgServer"]
