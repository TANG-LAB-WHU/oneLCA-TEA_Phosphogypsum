"""
Sensitivity Analysis Module

One-at-a-time and global sensitivity analysis.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class SensitivityResult:
    """Sensitivity analysis result."""

    parameter: str
    base_value: float
    sensitivity_coefficient: float  # % change in output / % change in input
    elasticity: float  # Normalized sensitivity
    importance_rank: int = 0


class SensitivityAnalyzer:
    """
    Sensitivity analysis for LCA-TEA models.

    Methods:
    - One-at-a-time (OAT)
    - Morris screening
    - Sobol indices (requires SALib)
    """

    def __init__(self, variation: float = 0.1):
        """
        Initialize analyzer.

        Args:
            variation: Fraction for OAT analysis (e.g., 0.1 = ±10%)
        """
        self.variation = variation

    def oat_analysis(
        self, parameters: Dict[str, float], calculation_func: Callable, output_name: str = "result"
    ) -> List[SensitivityResult]:
        """
        One-at-a-time sensitivity analysis.

        Args:
            parameters: Dict of parameter -> base value
            calculation_func: Function taking parameter dict
            output_name: Name of output to analyze
        """
        results = []

        # Calculate base result
        base_result = self._get_output(calculation_func(parameters), output_name)

        for param, base_value in parameters.items():
            if base_value == 0:
                continue

            # High value
            params_high = parameters.copy()
            params_high[param] = base_value * (1 + self.variation)
            result_high = self._get_output(calculation_func(params_high), output_name)

            # Low value
            params_low = parameters.copy()
            params_low[param] = base_value * (1 - self.variation)
            result_low = self._get_output(calculation_func(params_low), output_name)

            # Sensitivity coefficient
            delta_output = result_high - result_low
            delta_input = 2 * self.variation * base_value

            sensitivity = delta_output / delta_input if delta_input != 0 else 0

            # Elasticity (normalized)
            elasticity = (
                (delta_output / base_result) / (2 * self.variation) if base_result != 0 else 0
            )

            results.append(
                SensitivityResult(
                    parameter=param,
                    base_value=base_value,
                    sensitivity_coefficient=sensitivity,
                    elasticity=elasticity,
                )
            )

        # Rank by absolute elasticity
        results.sort(key=lambda x: abs(x.elasticity), reverse=True)
        for i, r in enumerate(results):
            r.importance_rank = i + 1

        return results

    def _get_output(self, result: Any, name: str) -> float:
        """Extract output value from result."""
        if isinstance(result, dict):
            return result.get(name, 0)
        return result

    def tornado_data(self, results: List[SensitivityResult], top_n: int = 10) -> Dict:
        """
        Prepare data for tornado plot.

        Args:
            results: Sensitivity results
            top_n: Number of parameters to include
        """
        top_results = results[:top_n]

        return {
            "parameters": [r.parameter for r in top_results],
            "elasticities": [r.elasticity for r in top_results],
            "base_values": [r.base_value for r in top_results],
            "directions": ["positive" if r.elasticity > 0 else "negative" for r in top_results],
        }

    def spider_plot_data(
        self,
        parameters: Dict[str, float],
        calculation_func: Callable,
        output_name: str = "result",
        variation_range: List[float] = None,
    ) -> Dict:
        """
        Generate spider plot data.

        Shows how output changes as each parameter varies.
        """
        variations = variation_range or [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]

        data = {"variations": variations, "parameters": {}}

        for param, base_value in parameters.items():
            outputs = []
            for var in variations:
                params_mod = parameters.copy()
                params_mod[param] = base_value * (1 + var)
                result = self._get_output(calculation_func(params_mod), output_name)
                outputs.append(result)
            data["parameters"][param] = outputs

        return data


def main():
    analyzer = SensitivityAnalyzer(variation=0.1)

    params = {
        "energy": 100,
        "efficiency": 0.9,
        "price": 50,
    }

    def calc(p):
        return {"cost": p["energy"] * p["price"] / p["efficiency"]}

    results = analyzer.oat_analysis(params, calc, "cost")

    for r in results:
        print(f"{r.parameter}: elasticity={r.elasticity:.3f}, rank={r.importance_rank}")


if __name__ == "__main__":
    main()
