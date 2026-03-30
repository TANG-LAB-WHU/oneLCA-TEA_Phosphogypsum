"""VAE smoke tests for stochastic_dynamics."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pgloop.stochastic_dynamics.trainer import train_vae
from pgloop.stochastic_dynamics.latent_sde import LatentSDE
from pgloop.stochastic_dynamics.vae import VAE


def test_vae_smoke_training_and_latent_dim():
    torch.set_default_dtype(torch.float64)
    x_np = np.random.default_rng(0).normal(size=(64, 4))
    x = torch.tensor(x_np, dtype=torch.float64)

    model = VAE(input_dim=4, latent_dim=3, hidden_dim=16)
    latent_sde = LatentSDE(latent_dim=3, hidden_dim=16)
    history = train_vae(
        model=model, x_data=x, latent_sde=latent_sde, n_epochs=5, lr=1e-3, beta=1e-2
    )
    recon, mu, _logvar, _z = model(x[:4])

    assert history["epochs"] == 5
    assert len(history["total"]) == 5
    assert len(history["fp"]) == 5
    assert recon.shape[-1] == 4
    assert mu.shape[-1] == 3

