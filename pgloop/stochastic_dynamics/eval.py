"""
Evaluation utilities for stochastic dynamics experiments.
"""

import time
from typing import Dict, List, Tuple

import numpy as np

# Version-compatible trapezoid function
trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def l2_pdf_error(p: np.ndarray, q: np.ndarray, dx: float) -> float:
    return float(np.sqrt(np.sum((p - q) ** 2) * dx))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def conservation_error(pdf: np.ndarray, dx: float) -> float:
    x = np.arange(pdf.size, dtype=float) * dx
    mass = float(trapezoid(pdf, x))
    return abs(mass - 1.0)


def stochastic_density_summary_from_timeseries(time_series_metrics: List[Dict]) -> Dict[str, float]:
    """
    Lightweight phase-4 parallel summary for integration in dynamic outputs.
    """
    if not time_series_metrics:
        return {}

    gwp = np.array(
        [pt.get("metrics", {}).get("gwp", 0.0) for pt in time_series_metrics],
        dtype=float,
    )
    clcc = np.array(
        [pt.get("metrics", {}).get("clcc", 0.0) for pt in time_series_metrics],
        dtype=float,
    )

    # A compact density-style diagnostic for phase-4 parallel reporting.
    return {
        "stochastic_density_entropy_proxy": float(np.mean(np.log1p(np.abs(gwp - np.mean(gwp))))),
        "stochastic_density_dispersion": float(np.std(clcc)),
        "stochastic_density_calibration_proxy": float(np.std(gwp) / (abs(np.mean(gwp)) + 1e-9)),
    }


def benchmark_callable(fn, *args, **kwargs) -> Tuple[float, object]:
    """
    Return elapsed seconds and function output.
    """
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return t1 - t0, out


def prediction_interval_coverage(
    y_true: np.ndarray, y_low: np.ndarray, y_high: np.ndarray
) -> float:
    covered = (y_true >= y_low) & (y_true <= y_high)
    return float(np.mean(covered))


def compare_with_numerical_solver(model, trajectory) -> dict:
    """
    Evaluates the PINN model on the space-time grid of a numerical trajectory
    and computes the relative L2 error:
      E = || p_PINN - p_numerical ||_2 / || p_numerical ||_2
    """
    import torch
    model.eval()

    nx = len(trajectory.x_grid)
    x_tensor = torch.tensor(trajectory.x_grid, dtype=torch.float64).unsqueeze(1)

    p_pinn_list = []
    with torch.no_grad():
        for t_val in trajectory.t_grid:
            t_tensor = torch.full((nx, 1), t_val, dtype=torch.float64)
            p_pred = model(x_tensor, t_tensor)
            p_pinn_list.append(p_pred.squeeze(-1).cpu().numpy())

    p_pinn = np.stack(p_pinn_list, axis=0)  # shape [nt, nx]
    p_num = trajectory.pdf_t

    # Compute L2 error
    l2_diff = np.sqrt(np.sum((p_pinn - p_num) ** 2))
    l2_num = np.sqrt(np.sum(p_num ** 2))

    rel_l2 = float(l2_diff / max(l2_num, 1e-9))
    return {
        "p_pinn": p_pinn,
        "rel_l2_error": rel_l2,
    }

