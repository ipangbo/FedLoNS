"""
Runtime compatibility helpers.

Some constrained environments expose a namespace-only NumPy package that does not
materialize the ``ndarray`` attribute eagerly. PyTorch expects ``numpy.ndarray`` to
exist during import, so we register a lightweight placeholder when necessary.
"""

from __future__ import annotations

import numpy as np


if not hasattr(np, "ndarray"):
    class _NumPyArrayPlaceholder:
        """Minimal placeholder class used when numpy.ndarray is unavailable."""

        pass

    np.ndarray = _NumPyArrayPlaceholder  # type: ignore[attr-defined]

