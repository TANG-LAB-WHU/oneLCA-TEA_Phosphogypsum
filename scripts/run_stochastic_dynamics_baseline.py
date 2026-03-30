"""
Run phase-4 stochastic-dynamics baseline:
finite-difference Fokker-Planck vs Monte Carlo histogram.
"""

import json
from pathlib import Path

import numpy as np

from pgloop.stochastic_dynamics.eval import conservation_error, kl_divergence, l2_pdf_error
from pgloop.stochastic_dynamics.fokker_planck import (
    FokkerPlanck1DSolver,
    const_diffusion,
    gaussian_pdf,
    monte_carlo_histogram,
    ou_drift,
)


def main():
    out_dir = Path("data/processed/dynamic_assessment")
    out_dir.mkdir(parents=True, exist_ok=True)

    solver = FokkerPlanck1DSolver(x_min=-5, x_max=5, n_x=301)
    p0 = gaussian_pdf(solver.x, mean=0.0, std=0.8)
    dt = 0.005
    n_steps = 200
    traj = solver.evolve(
        p0=p0,
        drift_fn=ou_drift(theta=1.2),
        diffusion_fn=const_diffusion(sigma=0.8),
        dt=dt,
        n_steps=n_steps,
    )

    centers, hist = monte_carlo_histogram(
        n_samples=30000,
        n_steps=n_steps,
        dt=dt,
        x0_std=0.8,
        drift_fn=ou_drift(theta=1.2),
        diffusion_fn=const_diffusion(sigma=0.8),
        bins=np.linspace(-5, 5, 301),
    )
    # interpolate to solver grid
    q = np.interp(solver.x, centers, hist, left=0.0, right=0.0)
    q = q / np.trapz(q, solver.x)
    p = traj.pdf_t[-1]

    metrics = {
        "l2_pdf": l2_pdf_error(p, q, solver.dx),
        "kl_p_q": kl_divergence(p, q),
        "mass_error": conservation_error(p, solver.dx),
        "n_steps": n_steps,
        "dt": dt,
    }
    with open(out_dir / "phase4_baseline_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

