"""
Adaptive Collocation Refinement (ACR) Sampler for PINN training.
"""

from typing import Callable, Tuple
import torch


class AdaptiveCollocationSampler:
    """
    Manages active collocation points in the spatio-temporal domain.
    Implements Residual-Based Adaptive Refinement (RAR) to inject points
    where the PDE residual is highest.
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        t_max: float,
        n_initial: int = 1000,
        device: str = "cpu",
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.t_max = t_max
        self.n_initial = n_initial
        self.device = device

        # Store active collocation points as PyTorch tensors
        self.x_points: torch.Tensor = torch.empty((0, 1), dtype=torch.float64, device=device)
        self.t_points: torch.Tensor = torch.empty((0, 1), dtype=torch.float64, device=device)

    def sample_initial(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate initial uniform random collocation points."""
        # Random uniform sample in [x_min, x_max]
        x = self.x_min + (self.x_max - self.x_min) * torch.rand(
            (self.n_initial, 1), dtype=torch.float64, device=self.device
        )
        # Random uniform sample in [0, t_max]
        t = self.t_max * torch.rand(
            (self.n_initial, 1), dtype=torch.float64, device=self.device
        )

        self.x_points = x
        self.t_points = t
        return self.x_points, self.t_points

    def refine_points(
        self,
        model: torch.nn.Module,
        drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        diffusion_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        n_new_points: int = 100,
        candidate_pool_size: int = 5000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate candidate points across the domain, compute their PDE residuals,
        and add the top n_new_points with the highest residuals to the active set.
        """
        if n_new_points <= 0:
            return self.x_points, self.t_points

        # 1. Generate candidate pool
        x_cand = self.x_min + (self.x_max - self.x_min) * torch.rand(
            (candidate_pool_size, 1), dtype=torch.float64, device=self.device
        )
        t_cand = self.t_max * torch.rand(
            (candidate_pool_size, 1), dtype=torch.float64, device=self.device
        )

        x_cand.requires_grad_(True)
        t_cand.requires_grad_(True)

        # 2. Compute residuals on candidates (no model update gradients, but keep graph for autograd)
        model.eval()
        # Keep track of grads, but we don't backprop on model weights
        res = model.residual(x_cand, t_cand, drift_fn, diffusion_fn)
        abs_res = torch.abs(res).detach()

        # 3. Sort candidates by residual
        _, top_indices = torch.topk(abs_res.squeeze(-1), k=n_new_points)

        # 4. Extract top candidates
        x_new = x_cand[top_indices].detach()
        t_new = t_cand[top_indices].detach()

        # 5. Append to active points
        self.x_points = torch.cat([self.x_points, x_new], dim=0)
        self.t_points = torch.cat([self.t_points, t_new], dim=0)

        model.train()
        return self.x_points, self.t_points
