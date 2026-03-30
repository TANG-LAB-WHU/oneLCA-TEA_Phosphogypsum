"""
Discernibility Analysis Module

Determines if differences between pathways are statistically significant.
"""

import numpy as np


class DiscernibilityAnalyzer:
    """Analyzes if pathway A is truly better than pathway B."""

    def __init__(self, mc_results_a: np.ndarray, mc_results_b: np.ndarray):
        """
        Initialize with Monte Carlo samples.

        Args:
            mc_results_a: Samples from pathway A
            mc_results_b: Samples from pathway B
        """
        self.samples_a = mc_results_a
        self.samples_b = mc_results_b

    def calculate_probability_a_better_than_b(self, target: str = "lower") -> float:
        """
        Calculate the probability that A is better than B.

        Args:
            target: "lower" if lower value is better (e.g. cost, GWP),
                    "higher" if higher is better (e.g. benefit)
        """
        if len(self.samples_a) != len(self.samples_b):
            # If unequal, we can only do random comparison or use distribution stats
            # Here we assume they correspond to the same MC iterations
            count = min(len(self.samples_a), len(self.samples_b))
            a = self.samples_a[:count]
            b = self.samples_b[:count]
        else:
            a, b = self.samples_a, self.samples_b

        if target == "lower":
            better = np.sum(a < b)
        else:
            better = np.sum(a > b)

        return float(better / len(a))

    def overlap_index(self) -> float:
        """Calculate the overlap between two distributions (0 to 1)."""
        # Simple implementation using histogram intersection
        combined = np.concatenate([self.samples_a, self.samples_b])
        bins = np.linspace(np.min(combined), np.max(combined), 50)

        hist_a, _ = np.histogram(self.samples_a, bins=bins, density=True)
        hist_b, _ = np.histogram(self.samples_b, bins=bins, density=True)

        return float(np.sum(np.minimum(hist_a, hist_b)) * (bins[1] - bins[0]))
