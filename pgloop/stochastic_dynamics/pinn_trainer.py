"""
Trainer for Stiff Boundary PINNs under Fokker-Planck Dynamics.
"""

import time
from typing import Callable, List, Optional
import torch

from pgloop.stochastic_dynamics.stiff_pinn import StiffBoundaryPINN
from pgloop.stochastic_dynamics.acr_sampler import AdaptiveCollocationSampler


class StiffPINNTrainer:
    """
    Two-phase trainer (Adam + L-BFGS) for Fokker-Planck PINN model,
    featuring adaptive collocation point refinement (ACR) and self-adaptive loss weighting.
    """

    def __init__(
        self,
        model: StiffBoundaryPINN,
        drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        diffusion_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_min: float,
        x_max: float,
        t_max: float,
        device: str = "cpu",
        bc_type: str = "no-flux",
        n_initial: int = 1200,
    ):
        self.model = model.to(device).double()
        self.drift_fn = drift_fn
        self.diffusion_fn = diffusion_fn
        self.x_min = x_min
        self.x_max = x_max
        self.t_max = t_max
        self.device = device
        self.bc_type = bc_type.lower()

        # Initialize the adaptive collocation sampler
        self.sampler = AdaptiveCollocationSampler(
            x_min=x_min,
            x_max=x_max,
            t_max=t_max,
            n_initial=n_initial,
            device=device,
        )

    def _compute_losses(
        self,
        x_coll: torch.Tensor,
        t_coll: torch.Tensor,
        p0_fn: Callable[[torch.Tensor], torch.Tensor],
        n_bc: int = 150,
        n_ic: int = 400,
        n_mass_t: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the PDE, boundary, initial, and mass conservation losses."""

        # 1. PDE loss
        residual = self.model.residual(x_coll, t_coll, self.drift_fn, self.diffusion_fn)
        loss_pde = torch.mean(residual**2)

        # 2. Boundary Condition (BC) loss
        # Sample points on the spatial boundary
        x_bc = torch.cat([
            torch.full((n_bc // 2, 1), self.x_min, dtype=torch.float64, device=self.device),
            torch.full((n_bc // 2, 1), self.x_max, dtype=torch.float64, device=self.device)
        ], dim=0).requires_grad_(True)
        t_bc = self.t_max * torch.rand((n_bc, 1), dtype=torch.float64, device=self.device).requires_grad_(True)

        if self.bc_type == "no-flux":
            p_bc = self.model(x_bc, t_bc)
            f_bc = self.drift_fn(x_bc, t_bc)
            g_bc = self.diffusion_fn(x_bc, t_bc)

            # zero flux condition: J = f*p - 0.5 * d_x(g^2 * p) = 0
            flux_diff = 0.5 * (g_bc * g_bc) * p_bc
            d_flux_diff_dx = torch.autograd.grad(
                flux_diff, x_bc, grad_outputs=torch.ones_like(flux_diff), create_graph=True
            )[0]
            flux = f_bc * p_bc - d_flux_diff_dx
            loss_bc = torch.mean(flux**2)
        else:
            # Dirichlet boundary condition: p(x_min, t) = p(x_max, t) = 0
            p_bc = self.model(x_bc, t_bc)
            loss_bc = torch.mean(p_bc**2)

        # 3. Initial Condition (IC) loss
        x_ic = self.x_min + (self.x_max - self.x_min) * torch.rand(
            (n_ic, 1), dtype=torch.float64, device=self.device
        )
        t_ic = torch.zeros((n_ic, 1), dtype=torch.float64, device=self.device)
        p_ic = self.model(x_ic, t_ic)
        p_ic_target = p0_fn(x_ic)
        loss_ic = torch.mean((p_ic - p_ic_target) ** 2)

        # 4. Mass conservation loss (integrate over x at multiple time slices)
        t_slices = self.t_max * torch.rand((n_mass_t, 1), dtype=torch.float64, device=self.device)
        x_grid = torch.linspace(self.x_min, self.x_max, 100, dtype=torch.float64, device=self.device).unsqueeze(1)
        dx = (self.x_max - self.x_min) / 99.0

        loss_mass_list = []
        for t_val in t_slices:
            t_expanded = t_val.expand_as(x_grid)
            p_grid = self.model(x_grid, t_expanded)
            # Trapezoidal integration in PyTorch
            integral = 0.5 * (p_grid[0] + p_grid[-1]) + torch.sum(p_grid[1:-1])
            integral = integral * dx
            loss_mass_list.append((integral - 1.0) ** 2)
        loss_mass = torch.mean(torch.stack(loss_mass_list))

        return loss_pde, loss_bc, loss_ic, loss_mass

    def train(
        self,
        p0_fn: Callable[[torch.Tensor], torch.Tensor],
        n_epochs_adam: int = 1500,
        n_epochs_lbfgs: int = 500,
        lr_adam: float = 1e-3,
        lr_lbfgs: float = 1.0,
        adaptive_sampling_freq: int = 150,
        n_new_collocation_points: int = 50,
        use_adaptive_weights: bool = True,
    ) -> dict:
        """
        Train the PINN using two phases: Adam followed by L-BFGS.
        Also performs ACR sampling and updates self-adaptive weights.
        """
        start_time = time.perf_counter()
        
        # 1. Sample initial collocation points
        x_coll, t_coll = self.sampler.sample_initial()

        # History trackers
        loss_history = []
        weight_history = []

        # List of parameters to optimize: model parameters AND adaptive weights (if enabled)
        params = list(self.model.parameters())
        optimizer_adam = torch.optim.Adam(params, lr=lr_adam)

        # Training Phase 1: Adam
        self.model.train()
        for epoch in range(1, n_epochs_adam + 1):
            optimizer_adam.zero_grad()

            loss_pde, loss_bc, loss_ic, loss_mass = self._compute_losses(
                x_coll, t_coll, p0_fn
            )

            if use_adaptive_weights:
                # Kendall-style multi-task learning loss balance:
                # L_total = exp(-log_w_i) * L_i + log_w_i
                total_loss = (
                    torch.exp(-self.model.log_w_pde) * loss_pde + self.model.log_w_pde +
                    torch.exp(-self.model.log_w_bc) * loss_bc + self.model.log_w_bc +
                    torch.exp(-self.model.log_w_ic) * loss_ic + self.model.log_w_ic +
                    torch.exp(-self.model.log_w_mass) * loss_mass + self.model.log_w_mass
                )
            else:
                # Fixed weight training
                total_loss = loss_pde + 1.0 * loss_bc + 1.0 * loss_ic + 0.1 * loss_mass

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer_adam.step()

            # Record loss history
            loss_history.append({
                "epoch": epoch,
                "total": total_loss.item(),
                "pde": loss_pde.item(),
                "bc": loss_bc.item(),
                "ic": loss_ic.item(),
                "mass": loss_mass.item(),
            })

            # Perform Adaptive Collocation Refinement
            if epoch % adaptive_sampling_freq == 0 and n_new_collocation_points > 0:
                x_coll, t_coll = self.sampler.refine_points(
                    self.model,
                    self.drift_fn,
                    self.diffusion_fn,
                    n_new_points=n_new_collocation_points,
                )

        # Training Phase 2: L-BFGS
        # We define L-BFGS only over model parameters to avoid instability with adaptive weights.
        # L-BFGS works best with model parameters alone, so we freeze the log_w parameters.
        model_params = [p for name, p in self.model.named_parameters() if "log_w" not in name]
        optimizer_lbfgs = torch.optim.LBFGS(
            model_params,
            lr=lr_lbfgs,
            max_iter=20,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer_lbfgs.zero_grad()
            l_pde, l_bc, l_ic, l_mass = self._compute_losses(x_coll, t_coll, p0_fn)
            
            # Use final frozen self-adaptive weights or fixed weights
            if use_adaptive_weights:
                loss = (
                    torch.exp(-self.model.log_w_pde).detach() * l_pde +
                    torch.exp(-self.model.log_w_bc).detach() * l_bc +
                    torch.exp(-self.model.log_w_ic).detach() * l_ic +
                    torch.exp(-self.model.log_w_mass).detach() * l_mass
                )
            else:
                loss = l_pde + 1.0 * l_bc + 1.0 * l_ic + 0.1 * l_mass
                
            loss.backward()
            return loss

        # Run L-BFGS for fixed steps/iterations
        for step in range(1, n_epochs_lbfgs + 1):
            loss_val = optimizer_lbfgs.step(closure)
            
            # Recompute individual losses for logging (requires autograd)
            l_pde, l_bc, l_ic, l_mass = self._compute_losses(x_coll, t_coll, p0_fn)
                
            loss_history.append({
                "epoch": n_epochs_adam + step,
                "total": loss_val.item(),
                "pde": l_pde.item(),
                "bc": l_bc.item(),
                "ic": l_ic.item(),
                "mass": l_mass.item(),
            })

        elapsed = time.perf_counter() - start_time
        
        return {
            "loss_history": loss_history,
            "elapsed_s": elapsed,
            "n_collocation_points": len(x_coll),
            "final_weights": self.model.get_weights() if use_adaptive_weights else None,
        }
