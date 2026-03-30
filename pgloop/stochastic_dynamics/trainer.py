"""
Training helpers for PINN and VAE stochastic dynamics experiments.
Includes optional checkpointing and JSON logging for reproducibility.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency
    torch = None
    F = None

from pgloop.stochastic_dynamics.losses import kl_loss, normalization_loss, pde_residual_loss, weighted_sum


def train_pinn(
    model,
    x_collocation,
    t_collocation,
    drift_fn,
    diffusion_fn,
    n_epochs: int = 200,
    lr: float = 1e-3,
    dx: float = 0.05,
    checkpoint_path: Optional[str] = None,
    log_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    if torch is None:
        raise ImportError("PyTorch is required for train_pinn.")

    model = model.double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: List[float] = []
    start = time.perf_counter()

    for epoch in range(n_epochs):
        opt.zero_grad()
        residual = model.residual(x_collocation, t_collocation, drift_fn, diffusion_fn)
        p = model.forward(x_collocation, t_collocation)
        loss_res = pde_residual_loss(residual)
        loss_norm = normalization_loss(p.view(1, -1), dx=dx)
        loss = weighted_sum(residual=loss_res, norm=loss_norm, w_residual=1.0, w_norm=0.1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        history.append(float(loss.detach().cpu().item()))

        if checkpoint_path and (epoch + 1 == n_epochs):
            ckpt = Path(checkpoint_path)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt)

    elapsed = time.perf_counter() - start
    payload = {
        "loss": history,
        "elapsed_s": elapsed,
        "epochs": n_epochs,
        "final_loss": history[-1] if history else 0.0,
    }
    if log_path:
        lp = Path(log_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        with open(lp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    return payload


def train_vae(
    model,
    x_data,
    latent_sde=None,
    n_epochs: int = 200,
    lr: float = 1e-3,
    beta: float = 1e-2,
    fp_weight: float = 0.05,
    checkpoint_path: Optional[str] = None,
    log_path: Optional[str] = None,
) -> Dict[str, List]:
    if torch is None:
        raise ImportError("PyTorch is required for train_vae.")

    model = model.double()
    x_data = x_data.double()
    params = list(model.parameters())
    if latent_sde is not None:
        latent_sde = latent_sde.double()
        params += list(latent_sde.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    recon_hist: List[float] = []
    total_hist: List[float] = []
    fp_hist: List[float] = []
    start = time.perf_counter()

    for epoch in range(n_epochs):
        opt.zero_grad()
        recon, mu, logvar, _z = model(x_data)
        recon_loss = F.mse_loss(recon, x_data)
        kl = kl_loss(mu, logvar, beta=beta)
        fp_loss = torch.tensor(0.0, dtype=torch.float64)
        if latent_sde is not None:
            # Simple FP-inspired regularization on latent drift/diffusion magnitude.
            t0 = torch.zeros((mu.shape[0], 1), dtype=mu.dtype, device=mu.device)
            drift = latent_sde.drift(mu, t0)
            diff = latent_sde.diffusion(mu, t0)
            fp_loss = torch.mean(drift**2) + torch.mean((diff - diff.mean(dim=0, keepdim=True)) ** 2)
        loss = recon_loss + kl + fp_weight * fp_loss
        loss.backward()
        opt.step()

        recon_hist.append(float(recon_loss.detach().cpu().item()))
        total_hist.append(float(loss.detach().cpu().item()))
        fp_hist.append(float(fp_loss.detach().cpu().item()))

        if checkpoint_path and (epoch + 1 == n_epochs):
            ckpt = Path(checkpoint_path)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "vae_state_dict": model.state_dict(),
                    "latent_sde_state_dict": latent_sde.state_dict() if latent_sde is not None else {},
                },
                ckpt,
            )

    elapsed = time.perf_counter() - start
    payload = {
        "recon": recon_hist,
        "total": total_hist,
        "fp": fp_hist,
        "elapsed_s": elapsed,
        "epochs": n_epochs,
    }
    if log_path:
        lp = Path(log_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        with open(lp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    return payload

