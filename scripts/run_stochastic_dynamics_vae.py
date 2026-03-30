"""
Run phase-4 VAE smoke training on synthetic trajectory-like data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyTorch is required for this script: {exc}")

from pgloop.stochastic_dynamics.latent_sde import LatentSDE
from pgloop.stochastic_dynamics.trainer import train_vae
from pgloop.stochastic_dynamics.vae import VAE


def main():
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(42)
    data_file = Path("data/processed/dynamic_assessment/dynamic_assessment_timeseries.csv")
    if data_file.exists():
        df = pd.read_csv(data_file)
        cols = [c for c in ["gwp", "clcc", "slcc", "lcop"] if c in df.columns]
        if len(cols) < 4:
            # fallback if some columns are missing
            x_data = rng.normal(loc=0.0, scale=1.0, size=(256, 4))
        else:
            arr = df[cols].dropna().to_numpy(dtype=float)
            # normalize for stable training
            arr = (arr - arr.mean(axis=0, keepdims=True)) / (arr.std(axis=0, keepdims=True) + 1e-9)
            x_data = arr
    else:
        # synthetic fallback for first-time run
        x_data = rng.normal(loc=0.0, scale=1.0, size=(256, 4))
    x = torch.tensor(x_data, dtype=torch.float64)

    model = VAE(input_dim=4, latent_dim=2, hidden_dim=32)
    latent_sde = LatentSDE(latent_dim=2, hidden_dim=32)
    history = train_vae(
        model,
        x_data=x,
        latent_sde=latent_sde,
        n_epochs=80,
        lr=2e-3,
        beta=5e-3,
        fp_weight=0.05,
        checkpoint_path="data/processed/dynamic_assessment/phase4_vae.ckpt",
        log_path="data/processed/dynamic_assessment/phase4_vae_log.json",
    )

    out_dir = Path("data/processed/dynamic_assessment")
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        _recon, mu, _logvar, _z = model(x[:32])
    payload = {
        "epochs": history["epochs"],
        "final_total_loss": history["total"][-1],
        "final_recon_loss": history["recon"][-1],
        "final_fp_loss": history["fp"][-1],
        "elapsed_s": history["elapsed_s"],
        "latent_dim": int(mu.shape[-1]),
        "data_source": "dynamic_assessment_timeseries.csv" if data_file.exists() else "synthetic",
    }
    with open(out_dir / "phase4_vae_smoke.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

