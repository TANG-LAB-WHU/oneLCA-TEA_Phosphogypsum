"""
Loss terms for stochastic-dynamics neural models.
"""

from typing import Optional

import torch


def pde_residual_loss(residual: torch.Tensor) -> torch.Tensor:
    return torch.mean(residual**2)


def boundary_loss(pred_boundary: torch.Tensor, target_boundary: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred_boundary - target_boundary) ** 2)


def normalization_loss(pdf: torch.Tensor, dx: float) -> torch.Tensor:
    mass = torch.sum(pdf, dim=-1) * dx
    return torch.mean((mass - 1.0) ** 2)


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    # Standard VAE KL(q(z|x)||N(0,I))
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return beta * kl


def weighted_sum(
    residual: Optional[torch.Tensor] = None,
    boundary: Optional[torch.Tensor] = None,
    norm: Optional[torch.Tensor] = None,
    recon: Optional[torch.Tensor] = None,
    kl: Optional[torch.Tensor] = None,
    w_residual: float = 1.0,
    w_boundary: float = 1.0,
    w_norm: float = 1.0,
    w_recon: float = 1.0,
    w_kl: float = 1.0,
) -> torch.Tensor:
    total = torch.tensor(0.0, dtype=torch.float64)
    if residual is not None:
        total = total + w_residual * residual
    if boundary is not None:
        total = total + w_boundary * boundary
    if norm is not None:
        total = total + w_norm * norm
    if recon is not None:
        total = total + w_recon * recon
    if kl is not None:
        total = total + w_kl * kl
    return total
