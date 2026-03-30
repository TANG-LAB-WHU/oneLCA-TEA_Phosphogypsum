"""
Joint uncertainty propagation for coupled LCA-TEA evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from pgloop.uncertainty.direct_sampling import MonteCarloSimulator


@dataclass
class JointPropagationResult:
    """Container for joint propagation samples and statistics."""

    n_samples: int
    parameter_samples: Dict[str, np.ndarray] = field(default_factory=dict)
    samples: Dict[str, np.ndarray] = field(default_factory=dict)
    summary: Dict[str, Dict[str, float]] = field(default_factory=dict)


class JointUncertaintyPropagator:
    """
    Synchronously propagates uncertainty through LCA and TEA engines.

    This ensures sampled parameters are shared between both domains,
    preserving cross-domain dependency in one Monte Carlo loop.
    """

    def __init__(self, lca_engine, tea_engine, n_iterations: int = 1000, seed: int = 42):
        self.lca_engine = lca_engine
        self.tea_engine = tea_engine
        self.n_iterations = n_iterations
        self.mc = MonteCarloSimulator(n_iterations=n_iterations, seed=seed)

    def propagate(
        self,
        pathway,
        functional_unit_value: float = 1.0,
        parameter_distributions: Optional[Dict[str, Dict]] = None,
        boundary_distributions: Optional[Dict[str, Dict]] = None,
    ) -> JointPropagationResult:
        """
        Run joint Monte Carlo propagation for one pathway.
        """
        distributions = parameter_distributions or pathway.get_parameter_distributions()
        boundary_distributions = boundary_distributions or {}
        all_distributions = {**distributions, **boundary_distributions}

        sampled = {
            name: self.mc.sample_from_spec(spec)
            for name, spec in all_distributions.items()
        }

        metrics = {
            "gwp": [],
            "clcc": [],
            "slcc": [],
            "lcop": [],
            "carbon_cost": [],
        }

        for i in range(self.n_iterations):
            params_i = {k: v[i] for k, v in sampled.items()}
            pathway_params = {k: val for k, val in params_i.items() if k in pathway.parameters}
            carbon_price = float(params_i.get("carbon_price_usd_t", 100.0))

            pathway_i = pathway.copy_with_parameters(pathway_params)

            lca_result = self.lca_engine.calculate(
                pathway_i, functional_unit_value=functional_unit_value, include_uncertainty=False
            )
            tea_result = self.tea_engine.calculate(
                pathway_i,
                functional_unit_value=functional_unit_value,
                include_external=True,
                include_uncertainty=False,
            )

            gwp = float(lca_result.impacts.get("climate_change", 0.0))
            clcc = float(tea_result.clcc)
            slcc = float(tea_result.slcc)
            carbon_cost = gwp * carbon_price / 1000.0

            metrics["gwp"].append(gwp)
            metrics["clcc"].append(clcc)
            metrics["slcc"].append(slcc)
            metrics["lcop"].append(self._estimate_lcop(pathway_i, clcc))
            metrics["carbon_cost"].append(carbon_cost)

        metric_arrays = {k: np.array(v, dtype=float) for k, v in metrics.items()}
        summary = {name: self._describe_samples(values) for name, values in metric_arrays.items()}
        return JointPropagationResult(
            n_samples=self.n_iterations,
            parameter_samples=sampled,
            samples=metric_arrays,
            summary=summary,
        )

    @staticmethod
    def _describe_samples(samples: np.ndarray) -> Dict[str, float]:
        """Summary stats used across dynamic assessment outputs."""
        return {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "p5": float(np.percentile(samples, 5)),
            "p50": float(np.percentile(samples, 50)),
            "p95": float(np.percentile(samples, 95)),
        }

    @staticmethod
    def _estimate_lcop(pathway, clcc: float) -> float:
        """
        Approximate levelized cost of phosphorus recovery.
        Falls back to CLCC when no explicit P-product is available.
        """
        products = pathway.get_products()
        phosphorus_output = 0.0
        for product in products:
            name = str(product.get("name", "")).lower()
            if "phosph" in name:
                phosphorus_output += float(product.get("quantity", 0.0))

        if phosphorus_output <= 0:
            return clcc
        return clcc / phosphorus_output

