"""
Multi-Criteria Decision Analysis (MCDA) Methods

TOPSIS, AHP, and Weighted Sum methods for pathway ranking.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from pgloop.decision.criteria import CriteriaSet, Direction


@dataclass
class MCDAResult:
    """Result of MCDA analysis."""

    rankings: List[str]  # Pathway names in order
    scores: Dict[str, float]  # Pathway -> score
    method: str

    def get_best(self) -> str:
        return self.rankings[0] if self.rankings else None

    def get_top_n(self, n: int) -> List[str]:
        return self.rankings[:n]


class WeightedSum:
    """
    Weighted Sum Method (WSM).

    Simple additive weighting: score = sum(w_i * v_i)
    """

    def __init__(self, criteria: CriteriaSet):
        self.criteria = criteria

    def rank(self, alternatives: Dict[str, Dict[str, float]]) -> MCDAResult:
        """
        Rank alternatives using weighted sum.

        Args:
            alternatives: {pathway_name: {criterion_name: value}}

        Returns:
            MCDAResult with rankings
        """
        scores = {}

        # Collect all values for normalization
        all_values = {c.name: [] for c in self.criteria.criteria}
        for alt_values in alternatives.values():
            for name, value in alt_values.items():
                if name in all_values:
                    all_values[name].append(value)

        for alt_name, alt_values in alternatives.items():
            score = 0.0

            for criterion in self.criteria.criteria:
                if criterion.name not in alt_values:
                    continue

                value = alt_values[criterion.name]

                # Normalize
                norm_value = criterion.normalize(value, all_values[criterion.name])

                # Adjust for direction (maximize: keep as is, minimize: invert)
                if criterion.direction == Direction.MINIMIZE:
                    norm_value = 1 - norm_value

                score += criterion.weight * norm_value

            scores[alt_name] = score

        # Sort by score (descending)
        rankings = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return MCDAResult(rankings=rankings, scores=scores, method="WeightedSum")


class TOPSIS:
    """
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).

    Ranks alternatives based on distance from ideal and anti-ideal solutions.
    """

    def __init__(self, criteria: CriteriaSet):
        self.criteria = criteria

    def rank(self, alternatives: Dict[str, Dict[str, float]]) -> MCDAResult:
        """
        Rank alternatives using TOPSIS.

        Args:
            alternatives: {pathway_name: {criterion_name: value}}

        Returns:
            MCDAResult with rankings
        """
        alt_names = list(alternatives.keys())
        crit_names = self.criteria.names
        n_alt = len(alt_names)
        n_crit = len(crit_names)

        if n_alt == 0 or n_crit == 0:
            return MCDAResult(rankings=[], scores={}, method="TOPSIS")

        # Build decision matrix
        matrix = np.zeros((n_alt, n_crit))
        for i, alt in enumerate(alt_names):
            for j, crit in enumerate(crit_names):
                matrix[i, j] = alternatives[alt].get(crit, 0)

        # Normalize (vector normalization)
        norms = np.linalg.norm(matrix, axis=0)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = matrix / norms

        # Apply weights
        weights = np.array([self.criteria.get_by_name(c).weight for c in crit_names])
        weighted = normalized * weights

        # Ideal and anti-ideal solutions
        ideal = np.zeros(n_crit)
        anti_ideal = np.zeros(n_crit)

        for j, crit_name in enumerate(crit_names):
            criterion = self.criteria.get_by_name(crit_name)
            col = weighted[:, j]

            if criterion.direction == Direction.MAXIMIZE:
                ideal[j] = np.max(col)
                anti_ideal[j] = np.min(col)
            else:
                ideal[j] = np.min(col)
                anti_ideal[j] = np.max(col)

        # Distance to ideal and anti-ideal
        dist_ideal = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
        dist_anti = np.sqrt(np.sum((weighted - anti_ideal) ** 2, axis=1))

        # TOPSIS score (closeness to ideal)
        topsis_scores = dist_anti / (dist_ideal + dist_anti + 1e-10)

        # Build result
        scores = {alt_names[i]: float(topsis_scores[i]) for i in range(n_alt)}
        rankings = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return MCDAResult(rankings=rankings, scores=scores, method="TOPSIS")


class AHP:
    """
    Analytic Hierarchy Process (simplified).

    Uses pairwise comparison for weight derivation.
    """

    def __init__(self, criteria: CriteriaSet):
        self.criteria = criteria

    def derive_weights_from_matrix(self, comparison_matrix: np.ndarray) -> np.ndarray:
        """
        Derive weights from pairwise comparison matrix.

        Uses eigenvalue method.
        """
        # Normalize columns
        col_sums = comparison_matrix.sum(axis=0)
        normalized = comparison_matrix / col_sums

        # Average rows (priority vector)
        weights = normalized.mean(axis=1)

        return weights

    def consistency_ratio(self, comparison_matrix: np.ndarray) -> float:
        """
        Calculate consistency ratio.

        CR < 0.1 is considered acceptable.
        """
        n = comparison_matrix.shape[0]

        # Random consistency index
        RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}

        # Calculate eigenvalue
        eigenvalues = np.linalg.eigvals(comparison_matrix)
        lambda_max = np.max(np.real(eigenvalues))

        # Consistency index
        CI = (lambda_max - n) / (n - 1) if n > 1 else 0

        # Consistency ratio
        CR = CI / RI.get(n, 1.45) if RI.get(n, 1.45) > 0 else 0

        return CR

    def rank(self, alternatives: Dict[str, Dict[str, float]]) -> MCDAResult:
        """
        Rank using AHP (uses predefined weights, not full pairwise comparison).

        For full AHP, use derive_weights_from_matrix first.
        """
        # Use WeightedSum as the synthesis step
        wsm = WeightedSum(self.criteria)
        result = wsm.rank(alternatives)
        result.method = "AHP"
        return result
