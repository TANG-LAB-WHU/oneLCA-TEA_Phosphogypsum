"""
Stochastic dynamics module.

Contains numerical and neural tools for density evolution:
- Fokker-Planck baselines
- PINN solvers
- VAE/latent stochastic dynamics
"""

from pgloop.stochastic_dynamics.eval import (
    benchmark_callable,
    conservation_error,
    kl_divergence,
    l2_pdf_error,
    phase4_density_summary_from_timeseries,
    prediction_interval_coverage,
)
from pgloop.stochastic_dynamics.fokker_planck import FokkerPlanck1DSolver, FokkerPlanck2DSolver

__all__ = [
    "FokkerPlanck1DSolver",
    "FokkerPlanck2DSolver",
    "l2_pdf_error",
    "kl_divergence",
    "conservation_error",
    "benchmark_callable",
    "prediction_interval_coverage",
    "phase4_density_summary_from_timeseries",
]

