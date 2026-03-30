"""
PINN model for 1D Fokker-Planck residual training.
"""

from typing import Callable, List

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


class FP_PINN(nn.Module):
    def __init__(self, hidden: List[int] = None):
        if torch is None:
            raise ImportError("PyTorch is required for FP_PINN.")
        super().__init__()
        hidden = hidden or [64, 64, 64]
        dims = [2] + hidden + [1]  # input: (x,t), output: p(x,t)
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, t], dim=-1)
        p = self.net(inp)
        # Softplus ensures non-negative density estimate
        return torch.nn.functional.softplus(p)

    def residual(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        diffusion_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        x.requires_grad_(True)
        t.requires_grad_(True)

        p = self.forward(x, t)
        p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]

        f = drift_fn(x, t)
        g = diffusion_fn(x, t)
        g2 = g * g

        fp = p_t + f * p_x - 0.5 * g2 * p_xx
        return fp

