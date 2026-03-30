"""
Evaluation utilities for stochastic dynamics experiments.
"""

import time
from typing import Dict, List, Tuple

import numpy as np


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
    mass = float(np.trapezoid(pdf, x))
    return abs(mass - 1.0)


def phase4_density_summary_from_timeseries(time_series_metrics: List[Dict]) -> Dict[str, float]:
    """
    Lightweight phase-4 parallel summary for integration in dynamic outputs.
    """
    if not time_series_metrics:
        return {}

    gwp = np.array([pt.get("metrics", {}).get("gwp", 0.0) for pt in time_series_metrics], dtype=float)
    clcc = np.array([pt.get("metrics", {}).get("clcc", 0.0) for pt in time_series_metrics], dtype=float)

    # A compact density-style diagnostic for phase-4 parallel reporting.
    return {
        "phase4_density_entropy_proxy": float(np.mean(np.log1p(np.abs(gwp - np.mean(gwp))))),
        "phase4_density_dispersion": float(np.std(clcc)),
        "phase4_density_calibration_proxy": float(np.std(gwp) / (abs(np.mean(gwp)) + 1e-9)),
    }


def benchmark_callable(fn, *args, **kwargs) -> Tuple[float, object]:
    """
    Return elapsed seconds and function output.
    """
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return t1 - t0, out


def prediction_interval_coverage(y_true: np.ndarray, y_low: np.ndarray, y_high: np.ndarray) -> float:
    covered = (y_true >= y_low) & (y_true <= y_high)
    return float(np.mean(covered))

