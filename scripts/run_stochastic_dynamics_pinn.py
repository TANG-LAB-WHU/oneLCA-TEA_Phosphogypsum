"""
Run phase-4 PINN smoke training for 1D Fokker-Planck residual.
"""

import json
from pathlib import Path

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyTorch is required for this script: {exc}")

from pgloop.stochastic_dynamics.eval import kl_divergence, l2_pdf_error
from pgloop.stochastic_dynamics.fokker_planck import (
    FokkerPlanck1DSolver,
    const_diffusion,
    gaussian_pdf,
    ou_drift,
)
from pgloop.stochastic_dynamics.pinn import FP_PINN
from pgloop.stochastic_dynamics.trainer import train_pinn


def main():
    torch.set_default_dtype(torch.float64)
    model = FP_PINN(hidden=[32, 32])

    x = torch.linspace(-2.0, 2.0, 128).reshape(-1, 1)
    t = torch.rand_like(x)

    def drift_fn(x_in, _t_in):
        return -1.0 * x_in

    def diffusion_fn(x_in, _t_in):
        return 0.7 * torch.ones_like(x_in)

    train_out = train_pinn(
        model=model,
        x_collocation=x,
        t_collocation=t,
        drift_fn=drift_fn,
        diffusion_fn=diffusion_fn,
        n_epochs=50,
        lr=2e-3,
        dx=float(x[1] - x[0]),
        checkpoint_path="data/processed/dynamic_assessment/stochastic_pinn.ckpt",
        log_path="data/processed/dynamic_assessment/stochastic_pinn_log.json",
    )

    # Compare PINN density at final time vs finite-difference baseline
    solver = FokkerPlanck1DSolver(x_min=-2.0, x_max=2.0, n_x=128)
    p0 = gaussian_pdf(solver.x, std=0.9)
    baseline = solver.evolve(
        p0=p0,
        drift_fn=ou_drift(theta=1.0),
        diffusion_fn=const_diffusion(sigma=0.7),
        dt=0.01,
        n_steps=50,
    )
    with torch.no_grad():
        x_eval = torch.tensor(solver.x.reshape(-1, 1), dtype=torch.float64)
        t_eval = torch.full_like(x_eval, 0.5)
        p_pred = model.forward(x_eval, t_eval).cpu().numpy().reshape(-1)
    p_pred = np.maximum(p_pred, 0.0)
    p_pred = p_pred / np.trapezoid(p_pred, solver.x)
    p_ref = baseline.pdf_t[-1]
    l2_err = l2_pdf_error(p_pred, p_ref, solver.dx)
    kl_err = kl_divergence(p_ref, p_pred)

    out_dir = Path("data/processed/dynamic_assessment")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epochs": train_out["epochs"],
        "final_loss": train_out["final_loss"],
        "elapsed_s": train_out["elapsed_s"],
        "l2_vs_baseline": l2_err,
        "kl_vs_baseline": kl_err,
        "history_head": train_out["loss"][:10],
    }
    with open(out_dir / "stochastic_pinn_smoke.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
