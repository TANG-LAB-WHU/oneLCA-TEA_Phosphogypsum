"""
Bayesian posterior update utilities for uncertainty priors.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from pgloop.uncertainty.propagation import JointUncertaintyPropagator
from pgloop.uncertainty.sensitivity import SensitivityAnalyzer


@dataclass
class PosteriorUpdateResult:
    """Posterior parameter summaries after Bayesian update."""

    priors: Dict[str, Dict]
    posterior: Dict[str, Dict]
    discrepancy: Dict[str, float] = field(default_factory=dict)


class BayesianUpdater:
    """
    Lightweight Bayesian updater with explicit discrepancy terms.
    """

    def __init__(self, observation_noise: float = 0.1):
        self.observation_noise = observation_noise

    def update_priors(
        self,
        priors: Dict[str, Dict],
        observations: Dict[str, float],
        predictions: Dict[str, float],
    ) -> PosteriorUpdateResult:
        """
        Update prior modes/means with Gaussian measurement model.
        """
        posterior = {}
        discrepancy = {}

        for name, spec in priors.items():
            prior_mean = self._spec_center(spec)
            prior_std = self._spec_std(spec)
            obs = float(observations.get(name, prior_mean))
            pred = float(predictions.get(name, prior_mean))
            eps = obs - pred
            discrepancy[name] = eps

            # Gaussian conjugate-style weighted update.
            obs_var = max(self.observation_noise**2, 1e-12)
            prior_var = max(prior_std**2, 1e-12)
            post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
            post_mean = post_var * (prior_mean / prior_var + obs / obs_var)
            post_std = float(np.sqrt(post_var))

            posterior[name] = self._updated_spec(spec, post_mean, post_std)

        return PosteriorUpdateResult(priors=priors, posterior=posterior, discrepancy=discrepancy)

    def run_closed_loop(
        self,
        pathway,
        lca_engine,
        tea_engine,
        priors: Dict[str, Dict],
        observations: Dict[str, float],
        n_iterations: int = 300,
    ) -> Dict[str, Dict]:
        """
        Closed-loop pass:
        propagate -> global sensitivity -> compare/update -> propagate.
        """
        propagator = JointUncertaintyPropagator(
            lca_engine=lca_engine,
            tea_engine=tea_engine,
            n_iterations=n_iterations,
            seed=42,
        )
        first_pass = propagator.propagate(pathway, boundary_distributions=priors)
        predictions = {name: stats["mean"] for name, stats in first_pass.summary.items()}
        global_sensitivity = self._compute_global_sensitivity(first_pass)

        updated = self.update_priors(priors=priors, observations=observations, predictions=predictions)
        second_pass = propagator.propagate(pathway, boundary_distributions=updated.posterior)

        return {
            "first_pass": first_pass.summary,
            "global_sensitivity": global_sensitivity,
            "posterior_priors": updated.posterior,
            "discrepancy": updated.discrepancy,
            "second_pass": second_pass.summary,
        }

    def _compute_global_sensitivity(self, first_pass) -> Dict[str, Dict]:
        """
        Compute Delta/Sobol-like indicators from first-pass samples.
        """
        if not first_pass.parameter_samples:
            return {}

        names = list(first_pass.parameter_samples.keys())
        X = np.column_stack([first_pass.parameter_samples[n] for n in names])
        Y = np.asarray(first_pass.samples.get("slcc", []), dtype=float)
        if Y.size == 0:
            return {}

        bounds = [[float(np.min(X[:, i])), float(np.max(X[:, i]))] for i in range(X.shape[1])]
        problem = {"num_vars": len(names), "names": names, "bounds": bounds}

        analyzer = SensitivityAnalyzer()
        output: Dict[str, Dict] = {}

        # Correlated-input-friendly analysis from existing sampled data.
        try:
            output["delta"] = analyzer.delta_analysis(problem=problem, X=X, Y=Y)
        except ImportError:
            output["delta"] = {}

        # Sobol on an inexpensive linear surrogate to keep runtime bounded.
        try:
            X_aug = np.column_stack([np.ones(X.shape[0]), X])
            beta, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)

            def surrogate_eval(X_new: np.ndarray) -> np.ndarray:
                Xn_aug = np.column_stack([np.ones(X_new.shape[0]), X_new])
                return Xn_aug @ beta

            output["sobol"] = analyzer.sobol_analysis(
                problem=problem,
                model_eval_fn=surrogate_eval,
                n_samples=min(256, max(64, X.shape[0] // 4)),
                calc_second_order=False,
            )
        except ImportError:
            output["sobol"] = {}

        return output

    @staticmethod
    def _spec_center(spec: Dict) -> float:
        if "mode" in spec:
            return float(spec["mode"])
        if "mean" in spec:
            return float(spec["mean"])
        if "value" in spec:
            return float(spec["value"])
        return 0.0

    @staticmethod
    def _spec_std(spec: Dict) -> float:
        if "std" in spec:
            return max(float(spec["std"]), 1e-6)
        if all(k in spec for k in ("min", "max")):
            # Approximate triangular/uniform spread.
            return max((float(spec["max"]) - float(spec["min"])) / 6.0, 1e-6)
        return 0.1

    @staticmethod
    def _updated_spec(spec: Dict, mean: float, std: float) -> Dict:
        kind = spec.get("type", "normal")
        if kind in {"normal", "lognormal"}:
            return {"type": kind, "mean": mean, "std": std}
        # For triangular/uniform priors, keep shape and shift center.
        span = std * 3.0
        return {
            "type": "triangular",
            "min": mean - span,
            "mode": mean,
            "max": mean + span,
        }

