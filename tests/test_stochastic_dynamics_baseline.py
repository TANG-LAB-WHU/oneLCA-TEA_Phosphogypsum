"""Tests for stochastic_dynamics Fokker-Planck baseline."""

import numpy as np

from pgloop.stochastic_dynamics.eval import conservation_error
from pgloop.stochastic_dynamics.fokker_planck import (
    FokkerPlanck1DSolver,
    FokkerPlanck2DSolver,
    const_diffusion,
    gaussian_pdf,
    ou_drift,
)


def test_fp_baseline_conservation_and_nonnegative():
    solver = FokkerPlanck1DSolver(x_min=-4.0, x_max=4.0, n_x=201)
    p0 = gaussian_pdf(solver.x, std=0.7)
    traj = solver.evolve(
        p0=p0,
        drift_fn=ou_drift(theta=1.0),
        diffusion_fn=const_diffusion(sigma=0.9),
        dt=0.005,
        n_steps=60,
    )
    p_final = traj.pdf_t[-1]
    assert np.all(p_final >= 0.0)
    assert conservation_error(p_final, solver.dx) < 1e-3


def test_fp_2d_baseline_runs():
    solver = FokkerPlanck2DSolver(x_min=-2, x_max=2, y_min=-2, y_max=2, n_x=41, n_y=41)
    p0 = np.exp(-0.5 * (solver.xx**2 + solver.yy**2))
    p0 = p0 / np.trapezoid(np.trapezoid(p0, solver.y, axis=1), solver.x)

    def drift_2d(xx, yy, _t):
        return -xx, -yy

    def diff_2d(xx, yy, _t):
        return 0.6 * np.ones_like(xx), 0.6 * np.ones_like(yy)

    traj = solver.evolve(p0=p0, drift_fn=drift_2d, diffusion_fn=diff_2d, dt=0.01, n_steps=10)
    p_final = traj.pdf_t[-1]
    assert np.all(p_final >= 0.0)
    mass = np.trapezoid(np.trapezoid(p_final, solver.y, axis=1), solver.x)
    assert abs(mass - 1.0) < 1e-3
