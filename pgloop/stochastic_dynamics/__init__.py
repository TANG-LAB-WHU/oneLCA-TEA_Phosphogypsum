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
    prediction_interval_coverage,
    stochastic_density_summary_from_timeseries,
)
from pgloop.stochastic_dynamics.fokker_planck import FokkerPlanck1DSolver, FokkerPlanck2DSolver

from pgloop.stochastic_dynamics.acr_sampler import AdaptiveCollocationSampler
from pgloop.stochastic_dynamics.pinn_trainer import StiffPINNTrainer
from pgloop.stochastic_dynamics.stiff_pinn import StiffBoundaryPINN

__all__ = [
    "FokkerPlanck1DSolver",
    "FokkerPlanck2DSolver",
    "StiffBoundaryPINN",
    "AdaptiveCollocationSampler",
    "StiffPINNTrainer",
    "l2_pdf_error",
    "kl_divergence",
    "conservation_error",
    "benchmark_callable",
    "prediction_interval_coverage",
    "stochastic_density_summary_from_timeseries",
]
