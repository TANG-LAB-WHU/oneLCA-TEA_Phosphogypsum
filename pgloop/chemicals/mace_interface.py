"""
MACE Model Interface

Provides a wrapper to initialize the MACE-MP machine learning force field
as an ASE calculator, automatically handling device placement (CPU/CUDA).
"""

import os
import torch
import warnings
from typing import Any

# Try importing mace
try:
    from mace.calculators import mace_mp
    MACE_AVAILABLE = True
except ImportError:
    mace_mp = None
    MACE_AVAILABLE = False


def is_mace_available() -> bool:
    """
    Check if the mace-torch dependency is available in the current environment.
    """
    return MACE_AVAILABLE


def get_mace_calculator(
    model_size: str = "medium",
    device: str = None,
    default_dtype: str = "float64",
) -> Any:
    """
    Load and return the MACE-MP foundation model calculator.

    Args:
        model_size (str): Size of the model to load ('small', 'medium', or 'large').
                          Defaults to 'medium'.
        device (str, optional): Target device ('cpu', 'cuda', etc.). If None, 
                                auto-detects based on CUDA availability.
        default_dtype (str): Default precision ('float32' or 'float64'). 
                             Defaults to 'float64' for better accuracy.

    Returns:
        ase.calculators.calculator.Calculator: The MACE calculator instance.
    """
    if not MACE_AVAILABLE:
        raise ImportError(
            "mace-torch is not installed or mace.calculators.mace_mp cannot be imported. "
            "Please install it using 'pip install mace-torch'."
        )

    # Device auto-detection
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Some Mac systems might try MPS, but MACE support on MPS can be unstable.
    # We explicitly log a warning if MPS is requested.
    if device == "mps":
        warnings.warn(
            "Running MACE on MPS (Apple Silicon GPU) might be unstable. "
            "Consider using 'cpu' if you encounter errors."
        )

    try:
        # Load MACE-MP foundation model calculator
        calculator = mace_mp(
            model=model_size,
            device=device,
            default_dtype=default_dtype,
        )
        return calculator
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MACE-MP model (size: {model_size}) on device {device}: {e}"
        ) from e
