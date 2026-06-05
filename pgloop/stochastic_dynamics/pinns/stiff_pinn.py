"""
Stiff Boundary PINN model for Fokker-Planck equations.
"""

from typing import Callable, List, Optional
import math

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class Sine(nn.Module):
    """Sine activation function with scaling factor for SIREN architectures."""
    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class StiffBoundaryPINN(nn.Module):
    """
    PINN architecture optimized for stiff boundary Fokker-Planck dynamics.
    Supports Siren and Tanh activations, learnable weights, and mass conservation.
    """

    def __init__(
        self,
        hidden: Optional[List[int]] = None,
        activation: str = "sine",
        w0: float = 30.0,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for StiffBoundaryPINN.")
        super().__init__()
        hidden = hidden or [64, 64, 64]
        self.activation_type = activation.lower()
        self.w0 = w0

        # Define layers
        dims = [2] + hidden + [1]
        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            linear = nn.Linear(in_dim, out_dim)
            
            # Custom initialization for SIREN (Sine) networks
            if self.activation_type == "sine":
                if i == 0:
                    # First layer: uniform in [-1/d_in, 1/d_in]
                    nn.init.uniform_(linear.weight, -1.0 / in_dim, 1.0 / in_dim)
                else:
                    # Hidden layers: uniform in [-sqrt(6/d_in)/w0, sqrt(6/d_in)/w0]
                    r = math.sqrt(6.0 / in_dim) / w0
                    nn.init.uniform_(linear.weight, -r, r)
                nn.init.zeros_(linear.bias)
                act = Sine(w0)
            else:
                # Standard Tanh initialization
                nn.init.xavier_normal_(linear.weight)
                nn.init.zeros_(linear.bias)
                act = nn.Tanh()

            layers.append(linear)
            layers.append(act)

        # Final output layer
        last_linear = nn.Linear(dims[-2], dims[-1])
        if self.activation_type == "sine":
            r = math.sqrt(6.0 / dims[-2]) / w0
            nn.init.uniform_(last_linear.weight, -r, r)
        else:
            nn.init.xavier_normal_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)
        layers.append(last_linear)

        self.net = nn.Sequential(*layers)

        # Learnable loss weights (log-scale to guarantee positivity)
        # Updated via backprop if learnable_weights=True during training
        self.log_w_pde = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.log_w_bc = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.log_w_ic = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.log_w_mass = nn.Parameter(torch.tensor(-2.3, dtype=torch.float64)) # starts around 0.1

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Output is forced to be non-negative using Softplus.
        x: [N, 1] tensor
        t: [N, 1] tensor
        """
        inp = torch.cat([x, t], dim=-1)
        p = self.net(inp)
        return torch.nn.functional.softplus(p)

    def residual(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        diffusion_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the Fokker-Planck PDE residual using automatic differentiation:
          R = p_t + d_x(f(x,t)p) - 0.5 * d_xx(g(x,t)^2 p)
        """
        # Ensure tracking gradients for inputs
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)

        p = self.forward(x, t)

        # 1. First-order time derivative: p_t
        p_t = torch.autograd.grad(
            p, t, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True
        )[0]

        # 2. Advection/drift term derivative: d_x(f(x,t) * p)
        f = drift_fn(x, t)
        adv_flux = f * p
        d_adv_dx = torch.autograd.grad(
            adv_flux, x, grad_outputs=torch.ones_like(adv_flux), create_graph=True, retain_graph=True
        )[0]

        # 3. Diffusion term second derivative: 0.5 * d_xx(g(x,t)^2 * p)
        g = diffusion_fn(x, t)
        diff_flux = 0.5 * (g * g) * p
        d_diff_dx = torch.autograd.grad(
            diff_flux, x, grad_outputs=torch.ones_like(diff_flux), create_graph=True, retain_graph=True
        )[0]
        d2_diff_dx2 = torch.autograd.grad(
            d_diff_dx, x, grad_outputs=torch.ones_like(d_diff_dx), create_graph=True, retain_graph=True
        )[0]

        # PDE residual
        res = p_t + d_adv_dx - d2_diff_dx2
        return res

    def get_weights(self) -> dict:
        """Return the positive loss weights."""
        return {
            "w_pde": torch.exp(self.log_w_pde).item(),
            "w_bc": torch.exp(self.log_w_bc).item(),
            "w_ic": torch.exp(self.log_w_ic).item(),
            "w_mass": torch.exp(self.log_w_mass).item(),
        }
