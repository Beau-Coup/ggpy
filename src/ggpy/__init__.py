# SPDX-FileCopyrightText: 2024-present ab <48172047+Beau-Coup@users.noreply.github.com>
#
# SPDX-License-Identifier: MIT
"""ggpy -- Gaussian processes on JAX aimed at real-time / online applications and as part of optimization objectives."""

from . import (
    _config,  # noqa: F401  (enables x64 before any array is created)
    fourier,
)
from .gp import GP, JITTER, MultiGP
from .kernel import (
    RBF,
    DiagKernel,
    Kernel,
    LinearTransformation,
    MultiKernel,
    NaiveDMatrix,
    RevLinear,
    SmallDMatrix,
    Stationary,
    Sum,
    WhiteNoise,
)
from .optimize import neg_log_likelihood, optimize

__all__ = [
    "GP",
    "fourier",
    "JITTER",
    "MultiGP",
    "RBF",
    "DiagKernel",
    "Kernel",
    "LinearTransformation",
    "MultiKernel",
    "NaiveDMatrix",
    "RevLinear",
    "SmallDMatrix",
    "Stationary",
    "Sum",
    "WhiteNoise",
    "neg_log_likelihood",
    "optimize",
]
