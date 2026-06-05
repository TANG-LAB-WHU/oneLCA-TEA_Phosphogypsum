from typing import Any, Callable, Dict, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class ReverseDesignOptimizer:
    """
    Reverse Design Optimizer using Bayesian Optimization.
    Searches the process parameter space to find input parameter sets
    that satisfy output performance bounds (e.g., carbon emissions, costs, NPV).
    """

    def __init__(
        self,
        evaluator_fn: Callable[[Dict[str, float]], Dict[str, float]],
        parameter_bounds: Dict[str, Tuple[float, float]],
        target_constraints: Dict[str, Dict[str, Any]],
        seed: int = 42,
    ):
        """
        Initialize the reverse design optimizer.

        Args:
            evaluator_fn: Callable mapping param_dict -> metric_dict
            parameter_bounds: Dict mapping param_name -> (min_val, max_val)
            target_constraints: Dict mapping metric_name -> {"type": "min"|"max", "value": float}
            seed: Random seed
        """
        self.evaluator_fn = evaluator_fn
        self.parameter_bounds = parameter_bounds
        self.target_constraints = target_constraints
        self.param_names = list(parameter_bounds.keys())
        self.rng = np.random.default_rng(seed)

        # Build bounds array
        self.bounds_arr = np.array([parameter_bounds[name] for name in self.param_names])

    def _calculate_utility(self, metrics: Dict[str, float]) -> float:
        """
        Calculate satisfaction utility of metrics.
        Returns 0.0 if all constraints are fully satisfied,
        and a negative penalty proportional to violation distance otherwise.
        """
        utility = 0.0
        for metric_name, constraint in self.target_constraints.items():
            val = metrics.get(metric_name)
            if val is None:
                continue

            target_val = constraint["value"]
            c_type = constraint["type"]

            if c_type == "min":
                if val < target_val:
                    # Quadratic penalty for violation
                    utility -= ((target_val - val) / (abs(target_val) + 1e-9)) ** 2
            elif c_type == "max":
                if val > target_val:
                    utility -= ((val - target_val) / (abs(target_val) + 1e-9)) ** 2
        return utility

    def _sample_random(self, n_samples: int) -> np.ndarray:
        """Draw random uniform samples within parameter bounds."""
        lows = self.bounds_arr[:, 0]
        highs = self.bounds_arr[:, 1]
        return self.rng.uniform(lows, highs, size=(n_samples, len(self.param_names)))

    def run(
        self,
        n_iterations: int = 20,
        n_initial_points: int = 5,
        kappa: float = 2.576,  # UCB exploration parameter
    ) -> Dict[str, Any]:
        """
        Run the Bayesian optimization loop.

        Args:
            n_iterations: Number of sequential query iterations
            n_initial_points: Number of initial random seed points
            kappa: Exploitation-exploration trade-off parameter for Upper Confidence Bound (UCB)

        Returns:
            Dict containing best parameters, history, and satisfaction status.
        """
        # 1. Evaluate initial random points
        x_train = self._sample_random(n_initial_points)
        y_train = []
        metrics_history = []

        for x in x_train:
            param_dict = dict(zip(self.param_names, x.tolist()))
            metrics = self.evaluator_fn(param_dict)
            utility = self._calculate_utility(metrics)
            y_train.append(utility)
            metrics_history.append((param_dict, metrics, utility))

        y_train = np.array(y_train)

        # 2. Bayesian Optimization Loop
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42,
        )

        for i in range(n_iterations):
            # Fit surrogate model
            gp.fit(x_train, y_train)

            # Generate dense set of candidate points to evaluate acquisition function
            candidates = self._sample_random(2000)
            mu, sigma = gp.predict(candidates, return_std=True)

            # Acquisition Function: Upper Confidence Bound (UCB)
            # We want to maximize utility (minimize penalty)
            ucb = mu + kappa * sigma
            best_idx = np.argmax(ucb)
            next_x = candidates[best_idx]

            # Evaluate next point
            param_dict = dict(zip(self.param_names, next_x.tolist()))
            metrics = self.evaluator_fn(param_dict)
            utility = self._calculate_utility(metrics)

            # Update training set
            x_train = np.vstack([x_train, next_x])
            y_train = np.append(y_train, utility)
            metrics_history.append((param_dict, metrics, utility))

        # Find best result
        best_idx = np.argmax(y_train)
        best_params, best_metrics, best_utility = metrics_history[best_idx]

        # Check if all constraints are fully met
        is_satisfied = best_utility == 0.0

        # Estimate parameter sensitivity/importance using surrogate correlation
        gp.fit(x_train, y_train)
        # Numerical gradient of surrogate model around the best point
        epsilon = 1e-5
        sensitivity = {}
        best_x_vec = np.array([best_params[name] for name in self.param_names])

        for idx, name in enumerate(self.param_names):
            x_plus = best_x_vec.copy()
            x_plus[idx] += epsilon
            x_minus = best_x_vec.copy()
            x_minus[idx] -= epsilon
            pred_plus = gp.predict(x_plus.reshape(1, -1))[0]
            pred_minus = gp.predict(x_minus.reshape(1, -1))[0]
            grad = (pred_plus - pred_minus) / (2.0 * epsilon)
            sensitivity[name] = float(grad)

        return {
            "best_parameters": best_params,
            "best_metrics": best_metrics,
            "best_utility": float(best_utility),
            "constraints_satisfied": is_satisfied,
            "parameter_sensitivities": sensitivity,
            "history": [
                {"parameters": h[0], "metrics": h[1], "utility": float(h[2])}
                for h in metrics_history
            ],
        }
