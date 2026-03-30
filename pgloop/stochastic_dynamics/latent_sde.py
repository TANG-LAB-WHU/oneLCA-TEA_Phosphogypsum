"""
Latent drift/diffusion parameterization utilities.
"""

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


class LatentSDE(nn.Module):
    """
    Neural drift and diffusion parameterization in latent space.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        if torch is None:
            raise ImportError("PyTorch is required for LatentSDE.")
        super().__init__()
        self.drift_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.diff_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus(),
        )

    def drift(self, z, t):
        inp = torch.cat([z, t], dim=-1)
        return self.drift_net(inp)

    def diffusion(self, z, t):
        inp = torch.cat([z, t], dim=-1)
        return self.diff_net(inp)

    def euler_maruyama_step(self, z, t, dt):
        drift = self.drift(z, t)
        diff = self.diffusion(z, t)
        noise = torch.randn_like(z)
        return z + drift * dt + diff * torch.sqrt(torch.tensor(dt, dtype=z.dtype)) * noise
