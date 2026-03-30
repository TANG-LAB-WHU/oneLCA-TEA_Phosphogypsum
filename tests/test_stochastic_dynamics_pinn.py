"""PINN smoke tests for stochastic_dynamics."""

import importlib.util

import pytest
import torch


def test_pinn_smoke_training_runs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed in test environment.")
    from pgloop.stochastic_dynamics.pinn import FP_PINN
    from pgloop.stochastic_dynamics.trainer import train_pinn

    torch.set_default_dtype(torch.float64)
    model = FP_PINN(hidden=[16, 16])
    x = torch.linspace(-1.5, 1.5, 64).reshape(-1, 1)
    t = torch.rand_like(x)

    def drift_fn(x_in, _t):
        return -x_in

    def diffusion_fn(x_in, _t):
        return 0.5 * torch.ones_like(x_in)

    out = train_pinn(
        model=model,
        x_collocation=x,
        t_collocation=t,
        drift_fn=drift_fn,
        diffusion_fn=diffusion_fn,
        n_epochs=5,
        lr=1e-3,
        dx=float(x[1] - x[0]),
    )
    assert out["epochs"] == 5
    assert len(out["loss"]) == 5
    assert out["final_loss"] >= 0.0
