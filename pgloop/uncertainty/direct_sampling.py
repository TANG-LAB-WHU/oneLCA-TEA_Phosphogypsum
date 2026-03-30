"""
Monte Carlo Simulation Module

Uncertainty propagation using Monte Carlo methods.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np


@dataclass
class MCResult:
    """Monte Carlo simulation result."""

    parameter: str
    n_iterations: int
    samples: np.ndarray
    mean: float
    std: float
    cv: float
    percentiles: Dict[int, float]


class MonteCarloSimulator:
    """
    Monte Carlo simulator for uncertainty propagation.

    Supports:
    - Triangular distributions
    - Normal distributions
    - Uniform distributions
    - Log-normal distributions
    """

    def __init__(self, n_iterations: int = 1000, seed: int = None):
        self.n_iterations = n_iterations
        if seed is not None:
            np.random.seed(seed)

    def sample_triangular(self, low: float, mode: float, high: float, n: int = None) -> np.ndarray:
        """Sample from triangular distribution."""
        return np.random.triangular(low, mode, high, n or self.n_iterations)

    def sample_normal(self, mean: float, std: float, n: int = None) -> np.ndarray:
        """Sample from normal distribution."""
        return np.random.normal(mean, std, n or self.n_iterations)

    def sample_uniform(self, low: float, high: float, n: int = None) -> np.ndarray:
        """Sample from uniform distribution."""
        return np.random.uniform(low, high, n or self.n_iterations)

    def sample_lognormal(self, mean: float, sigma: float, n: int = None) -> np.ndarray:
        """Sample from log-normal distribution."""
        return np.random.lognormal(mean, sigma, n or self.n_iterations)

    def sample_from_spec(self, spec: Dict, n: int = None) -> np.ndarray:
        """
        Sample from distribution specification.

        Args:
            spec: Dict with 'type' and distribution parameters
        """
        n = n or self.n_iterations
        dist_type = spec.get("type", "triangular")

        if dist_type == "triangular":
            return self.sample_triangular(spec["min"], spec["mode"], spec["max"], n)
        elif dist_type == "normal":
            return self.sample_normal(spec["mean"], spec["std"], n)
        elif dist_type == "uniform":
            return self.sample_uniform(spec["min"], spec["max"], n)
        elif dist_type == "lognormal":
            return self.sample_lognormal(spec["mean"], spec["sigma"], n)
        elif dist_type == "fixed":
            return np.full(n, spec.get("value", spec.get("mode", 0)))
        else:
            raise ValueError(f"Unknown distribution: {dist_type}")

    def propagate(
        self,
        distributions: Dict[str, Dict],
        calculation_func: Callable,
        output_names: List[str] = None,
    ) -> Dict[str, MCResult]:
        """
        Propagate uncertainty through a calculation.

        Args:
            distributions: Dict mapping parameter to distribution spec
            calculation_func: Function that takes sampled parameters
            output_names: Names of output variables
        """
        # Sample all parameters
        sampled_params = {}
        for param, spec in distributions.items():
            sampled_params[param] = self.sample_from_spec(spec)

        # Run calculations
        results = []
        for i in range(self.n_iterations):
            params_i = {k: v[i] for k, v in sampled_params.items()}
            try:
                result = calculation_func(params_i)
                results.append(result)
            except Exception:
                results.append(None)

        # Filter out failed runs
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            return {}

        # Process results
        if isinstance(valid_results[0], dict):
            # Multiple outputs
            output_names = output_names or list(valid_results[0].keys())
            mc_results = {}

            for name in output_names:
                samples = np.array([r[name] for r in valid_results if name in r])
                mc_results[name] = self._compute_statistics(name, samples)

            return mc_results
        else:
            # Single output
            samples = np.array(valid_results)
            return {"result": self._compute_statistics("result", samples)}

    def _compute_statistics(self, name: str, samples: np.ndarray) -> MCResult:
        """Compute statistics from samples."""
        mean = float(np.mean(samples))
        std = float(np.std(samples))
        cv = std / mean if mean != 0 else 0

        percentiles = {
            5: float(np.percentile(samples, 5)),
            25: float(np.percentile(samples, 25)),
            50: float(np.percentile(samples, 50)),
            75: float(np.percentile(samples, 75)),
            95: float(np.percentile(samples, 95)),
        }

        return MCResult(
            parameter=name,
            n_iterations=len(samples),
            samples=samples,
            mean=mean,
            std=std,
            cv=cv,
            percentiles=percentiles,
        )


def main():
    mc = MonteCarloSimulator(n_iterations=1000)

    distributions = {
        "energy": {"type": "triangular", "min": 80, "mode": 100, "max": 120},
        "efficiency": {"type": "normal", "mean": 0.9, "std": 0.05},
    }

    def calc(params):
        return {"output": params["energy"] / params["efficiency"]}

    results = mc.propagate(distributions, calc)
    print(f"Mean output: {results['output'].mean:.2f}")
    print(
        f"95% CI: [{results['output'].percentiles[5]:.2f}, {results['output'].percentiles[95]:.2f}]"
    )


if __name__ == "__main__":
    main()
