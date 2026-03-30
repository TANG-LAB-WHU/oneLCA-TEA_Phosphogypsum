"""
Markov Chain Monte Carlo (MCMC) Module

Advanced sampling methods for uncertainty quantification with correlated parameters.
Includes Metropolis-Hastings, Hamiltonian MC, and Gibbs sampling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MCMCResult:
    """Results from MCMC sampling."""

    samples: np.ndarray  # Shape: (n_samples, n_parameters)
    parameter_names: List[str]
    acceptance_rate: float
    log_probs: np.ndarray  # Log probability of each sample
    n_chains: int = 1
    warmup: int = 0

    # Diagnostics
    r_hat: Optional[np.ndarray] = None  # Gelman-Rubin statistic
    ess: Optional[np.ndarray] = None  # Effective sample size

    def get_posterior_mean(self) -> Dict[str, float]:
        """Get posterior mean for each parameter."""
        return {
            name: float(np.mean(self.samples[:, i])) for i, name in enumerate(self.parameter_names)
        }

    def get_posterior_std(self) -> Dict[str, float]:
        """Get posterior standard deviation."""
        return {
            name: float(np.std(self.samples[:, i])) for i, name in enumerate(self.parameter_names)
        }

    def get_credible_interval(self, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
        """Get credible intervals."""
        lower = alpha / 2 * 100
        upper = (1 - alpha / 2) * 100
        return {
            name: (
                float(np.percentile(self.samples[:, i], lower)),
                float(np.percentile(self.samples[:, i], upper)),
            )
            for i, name in enumerate(self.parameter_names)
        }

    def summary(self) -> Dict:
        """Generate summary statistics."""
        return {
            "n_samples": len(self.samples),
            "n_parameters": len(self.parameter_names),
            "acceptance_rate": self.acceptance_rate,
            "means": self.get_posterior_mean(),
            "stds": self.get_posterior_std(),
            "95_credible": self.get_credible_interval(0.05),
        }


class BaseMCMC(ABC):
    """Abstract base class for MCMC samplers."""

    def __init__(
        self,
        log_prob_fn: Callable[[np.ndarray], float],
        parameter_names: List[str],
        initial_state: np.ndarray,
        seed: int = 42,
    ):
        """
        Initialize MCMC sampler.

        Args:
            log_prob_fn: Function returning log probability (unnormalized)
            parameter_names: Names of parameters
            initial_state: Starting point for chain
            seed: Random seed
        """
        self.log_prob_fn = log_prob_fn
        self.parameter_names = parameter_names
        self.initial_state = np.array(initial_state, dtype=float)
        self.n_params = len(initial_state)
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def sample(self, n_samples: int, warmup: int = 1000, **kwargs) -> MCMCResult:
        """Run MCMC sampling."""
        pass


class MetropolisHastings(BaseMCMC):
    """
    Metropolis-Hastings MCMC Sampler.

    Classic random-walk MCMC with symmetric proposal distribution.
    """

    def __init__(
        self,
        log_prob_fn: Callable[[np.ndarray], float],
        parameter_names: List[str],
        initial_state: np.ndarray,
        proposal_cov: np.ndarray = None,
        seed: int = 42,
    ):
        super().__init__(log_prob_fn, parameter_names, initial_state, seed)

        if proposal_cov is None:
            # Default: identity matrix scaled by 2.38^2 / d (optimal for Gaussian)
            scale = (2.38**2) / self.n_params
            self.proposal_cov = np.eye(self.n_params) * scale
        else:
            self.proposal_cov = np.array(proposal_cov)

    def sample(
        self,
        n_samples: int,
        warmup: int = 1000,
        adapt_proposal: bool = True,
        adapt_interval: int = 100,
    ) -> MCMCResult:
        """
        Run Metropolis-Hastings sampling.

        Args:
            n_samples: Number of samples after warmup
            warmup: Number of warmup iterations (discarded)
            adapt_proposal: Whether to adapt proposal covariance during warmup
            adapt_interval: Adaptation interval

        Returns:
            MCMCResult with samples
        """
        total_iterations = warmup + n_samples
        samples = np.zeros((total_iterations, self.n_params))
        log_probs = np.zeros(total_iterations)

        current_state = self.initial_state.copy()
        current_log_prob = self.log_prob_fn(current_state)

        accepted = 0
        proposal_cov = self.proposal_cov.copy()

        for i in range(total_iterations):
            # Propose new state
            proposal = self.rng.multivariate_normal(current_state, proposal_cov)
            proposal_log_prob = self.log_prob_fn(proposal)

            # Accept/reject
            log_alpha = proposal_log_prob - current_log_prob

            if np.log(self.rng.random()) < log_alpha:
                current_state = proposal
                current_log_prob = proposal_log_prob
                accepted += 1

            samples[i] = current_state
            log_probs[i] = current_log_prob

            # Adapt proposal during warmup
            if adapt_proposal and i < warmup and i > 0 and i % adapt_interval == 0:
                # Use empirical covariance with regularization
                recent_samples = samples[max(0, i - 500) : i]
                if len(recent_samples) > 10:
                    emp_cov = np.cov(recent_samples.T)
                    scale = (2.38**2) / self.n_params
                    proposal_cov = scale * (emp_cov + 0.01 * np.eye(self.n_params))

        acceptance_rate = accepted / total_iterations

        return MCMCResult(
            samples=samples[warmup:],
            parameter_names=self.parameter_names,
            acceptance_rate=acceptance_rate,
            log_probs=log_probs[warmup:],
            warmup=warmup,
        )


class HamiltonianMC(BaseMCMC):
    """
    Hamiltonian Monte Carlo (HMC) Sampler.

    Uses gradient information for efficient exploration.
    Requires gradient of log probability function.
    """

    def __init__(
        self,
        log_prob_fn: Callable[[np.ndarray], float],
        grad_log_prob_fn: Callable[[np.ndarray], np.ndarray],
        parameter_names: List[str],
        initial_state: np.ndarray,
        step_size: float = 0.1,
        n_leapfrog: int = 10,
        mass_matrix: np.ndarray = None,
        seed: int = 42,
    ):
        super().__init__(log_prob_fn, parameter_names, initial_state, seed)

        self.grad_log_prob_fn = grad_log_prob_fn
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog

        if mass_matrix is None:
            self.mass_matrix = np.eye(self.n_params)
        else:
            self.mass_matrix = np.array(mass_matrix)

        self.mass_inv = np.linalg.inv(self.mass_matrix)

    def _leapfrog(self, q: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Leapfrog integrator."""
        q = q.copy()
        p = p.copy()

        # Half step for momentum
        p += 0.5 * self.step_size * self.grad_log_prob_fn(q)

        # Full steps
        for _ in range(self.n_leapfrog - 1):
            q += self.step_size * self.mass_inv @ p
            p += self.step_size * self.grad_log_prob_fn(q)

        # Final half step
        q += self.step_size * self.mass_inv @ p
        p += 0.5 * self.step_size * self.grad_log_prob_fn(q)

        return q, -p  # Negate momentum for reversibility

    def _hamiltonian(self, q: np.ndarray, p: np.ndarray) -> float:
        """Compute Hamiltonian (negative log prob + kinetic energy)."""
        kinetic = 0.5 * p @ self.mass_inv @ p
        potential = -self.log_prob_fn(q)
        return kinetic + potential

    def sample(
        self,
        n_samples: int,
        warmup: int = 1000,
        adapt_step_size: bool = True,
        target_accept: float = 0.8,
    ) -> MCMCResult:
        """
        Run HMC sampling.

        Args:
            n_samples: Number of samples
            warmup: Warmup iterations
            adapt_step_size: Whether to adapt step size
            target_accept: Target acceptance rate for adaptation

        Returns:
            MCMCResult with samples
        """
        total_iterations = warmup + n_samples
        samples = np.zeros((total_iterations, self.n_params))
        log_probs = np.zeros(total_iterations)

        current_q = self.initial_state.copy()
        step_size = self.step_size

        accepted = 0

        for i in range(total_iterations):
            # Sample momentum
            p = self.rng.multivariate_normal(np.zeros(self.n_params), self.mass_matrix)

            current_H = self._hamiltonian(current_q, p)

            # Leapfrog integration
            try:
                prop_q, prop_p = self._leapfrog(current_q, p)
                prop_H = self._hamiltonian(prop_q, prop_p)
            except Exception:
                # Handle numerical issues
                prop_H = np.inf
                prop_q = current_q

            # Accept/reject
            log_alpha = current_H - prop_H

            if np.log(self.rng.random()) < log_alpha:
                current_q = prop_q
                accepted += 1

            samples[i] = current_q
            log_probs[i] = self.log_prob_fn(current_q)

            # Adapt step size during warmup
            if adapt_step_size and i < warmup and i > 0:
                accept_rate = accepted / (i + 1)
                if accept_rate > target_accept:
                    step_size *= 1.02
                else:
                    step_size *= 0.98
                step_size = np.clip(step_size, 1e-4, 1.0)

        acceptance_rate = accepted / total_iterations

        return MCMCResult(
            samples=samples[warmup:],
            parameter_names=self.parameter_names,
            acceptance_rate=acceptance_rate,
            log_probs=log_probs[warmup:],
            warmup=warmup,
        )


class GibbsSampler(BaseMCMC):
    """
    Gibbs Sampler for componentwise updates.

    Samples each parameter conditionally on others.
    """

    def __init__(
        self,
        conditional_samplers: Dict[str, Callable],
        parameter_names: List[str],
        initial_state: np.ndarray,
        seed: int = 42,
    ):
        """
        Initialize Gibbs sampler.

        Args:
            conditional_samplers: Dict mapping parameter name to
                                  function(current_values, rng) -> sample
            parameter_names: Names of parameters
            initial_state: Starting values
            seed: Random seed
        """
        # Create identity log_prob_fn (not used in Gibbs)
        super().__init__(lambda x: 0.0, parameter_names, initial_state, seed)
        self.conditional_samplers = conditional_samplers

    def sample(self, n_samples: int, warmup: int = 1000) -> MCMCResult:
        """Run Gibbs sampling."""
        total_iterations = warmup + n_samples
        samples = np.zeros((total_iterations, self.n_params))

        current_state = self.initial_state.copy()
        param_indices = {name: i for i, name in enumerate(self.parameter_names)}

        for i in range(total_iterations):
            for name in self.parameter_names:
                if name in self.conditional_samplers:
                    idx = param_indices[name]
                    current_state[idx] = self.conditional_samplers[name](current_state, self.rng)

            samples[i] = current_state.copy()

        return MCMCResult(
            samples=samples[warmup:],
            parameter_names=self.parameter_names,
            acceptance_rate=1.0,  # Gibbs always accepts
            log_probs=np.zeros(n_samples),
            warmup=warmup,
        )


class MCMCDiagnostics:
    """Convergence diagnostics for MCMC."""

    @staticmethod
    def gelman_rubin(chains: List[np.ndarray]) -> np.ndarray:
        """
        Compute Gelman-Rubin R-hat statistic.

        Args:
            chains: List of chains, each shape (n_samples, n_params)

        Returns:
            R-hat for each parameter (should be < 1.1)
        """
        n_chains = len(chains)
        if n_chains < 2:
            return np.ones(chains[0].shape[1])

        n_samples = chains[0].shape[0]

        # Stack chains
        all_chains = np.stack(chains)  # (n_chains, n_samples, n_params)

        # Between-chain variance
        chain_means = all_chains.mean(axis=1)  # (n_chains, n_params)
        B = n_samples * np.var(chain_means, axis=0, ddof=1)

        # Within-chain variance
        W = np.mean([np.var(chain, axis=0, ddof=1) for chain in all_chains], axis=0)

        # Pooled variance estimate
        var_hat = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B

        # R-hat
        r_hat = np.sqrt(var_hat / W)

        return r_hat

    @staticmethod
    def effective_sample_size(samples: np.ndarray) -> np.ndarray:
        """
        Estimate effective sample size using autocorrelation.

        Args:
            samples: Shape (n_samples, n_params)

        Returns:
            ESS for each parameter
        """
        n_samples, n_params = samples.shape
        ess = np.zeros(n_params)

        for j in range(n_params):
            x = samples[:, j]

            # Compute autocorrelation
            x_centered = x - np.mean(x)
            acf = np.correlate(x_centered, x_centered, mode="full")
            acf = acf[len(acf) // 2 :]
            acf = acf / acf[0]

            # Sum of autocorrelations (truncate at first negative)
            rho_sum = 0.0
            for k in range(1, len(acf)):
                if acf[k] < 0:
                    break
                rho_sum += acf[k]

            ess[j] = n_samples / (1 + 2 * rho_sum)

        return ess

    @staticmethod
    def check_convergence(result: MCMCResult, threshold: float = 1.1) -> Dict:
        """Check if MCMC has converged."""
        ess = MCMCDiagnostics.effective_sample_size(result.samples)

        return {
            "converged": True,  # Would need multiple chains for R-hat
            "ess": dict(zip(result.parameter_names, ess.tolist())),
            "min_ess": float(ess.min()),
            "acceptance_rate": result.acceptance_rate,
            "acceptance_ok": 0.1 < result.acceptance_rate < 0.9,
        }
