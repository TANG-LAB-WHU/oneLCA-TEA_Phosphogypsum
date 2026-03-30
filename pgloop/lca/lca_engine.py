"""
LCA Engine Module

Main calculation engine for Life Cycle Assessment.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pgloop.lca.characterization import CharacterizationFactors
from pgloop.lca.impact_assessment import ImpactAssessment


@dataclass
class LCAResult:
    """Complete LCA result for a treatment pathway."""

    pathway_code: str
    functional_unit: str
    functional_unit_value: float
    inventory: Dict[str, List[Dict]]
    impacts: Dict[str, float]
    normalized_impacts: Dict[str, float]
    uncertainty: Dict[str, Dict]
    metadata: Dict = field(default_factory=dict)


class LCAEngine:
    """
    Life Cycle Assessment calculation engine.

    Implements ISO 14040/14044 methodology:
    1. Goal and Scope Definition
    2. Life Cycle Inventory (LCI) Analysis
    3. Life Cycle Impact Assessment (LCIA)
    4. Interpretation

    Functional Unit: 1 tonne phosphogypsum treated
    """

    # Default functional unit
    FUNCTIONAL_UNIT = "1 tonne phosphogypsum treated"
    FUNCTIONAL_UNIT_VALUE = 1000  # kg

    # Impact categories
    IMPACT_CATEGORIES = [
        "climate_change",  # kg CO2-eq
        "acidification",  # mol H+-eq
        "eutrophication_fresh",  # kg P-eq
        "eutrophication_marine",  # kg N-eq
        "human_toxicity_cancer",  # CTUh
        "human_toxicity_noncancer",  # CTUh
        "ecotoxicity_freshwater",  # CTUe
        "ionizing_radiation",  # kBq U-235 eq
        "particulate_matter",  # disease incidence
        "resource_depletion",  # kg Sb-eq
    ]

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the LCA engine.

        Args:
            config_path: Path to configuration files
        """
        self.config_path = config_path or Path("./config")
        self.characterization = CharacterizationFactors(self.config_path)
        self.impact_assessment = ImpactAssessment(self.characterization)

    def calculate(
        self,
        pathway,  # Treatment pathway instance
        functional_unit_value: float = 1.0,  # tonnes
        include_uncertainty: bool = True,
    ) -> LCAResult:
        """
        Calculate LCA for a treatment pathway.

        Args:
            pathway: Treatment pathway instance with LCI data
            functional_unit_value: Amount in tonnes (default 1)
            include_uncertainty: Whether to calculate uncertainty

        Returns:
            LCAResult with complete assessment
        """
        # Scale inventory to functional unit
        inventory = pathway.get_scaled_inventory(functional_unit_value * 1000)  # Convert to kg

        # Calculate impacts
        impacts = self.impact_assessment.calculate(inventory)

        # Normalize impacts (optional)
        normalized = self.impact_assessment.normalize(impacts)

        # Uncertainty analysis
        uncertainty = {}
        if include_uncertainty:
            uncertainty = self._calculate_uncertainty(pathway, functional_unit_value)

        return LCAResult(
            pathway_code=pathway.code,
            functional_unit=self.FUNCTIONAL_UNIT,
            functional_unit_value=functional_unit_value,
            inventory=inventory,
            impacts=impacts,
            normalized_impacts=normalized,
            uncertainty=uncertainty,
            metadata={
                "country": getattr(pathway, "country", "global"),
                "year": getattr(pathway, "year", 2024),
            },
        )

    def compare_pathways(
        self, pathways: List, functional_unit_value: float = 1.0
    ) -> Dict[str, LCAResult]:
        """
        Compare multiple treatment pathways.

        Args:
            pathways: List of pathway instances
            functional_unit_value: Amount in tonnes

        Returns:
            Dict mapping pathway code to LCA result
        """
        results = {}
        for pathway in pathways:
            results[pathway.code] = self.calculate(pathway, functional_unit_value)
        return results

    def _calculate_uncertainty(
        self, pathway, functional_unit_value: float, n_iterations: int = 1000
    ) -> Dict[str, Dict]:
        """
        Calculate uncertainty using Monte Carlo simulation.

        Returns:
            Dict with mean, std, percentiles for each impact category
        """
        uncertainty = {}

        # Get parameter distributions from pathway
        distributions = pathway.get_parameter_distributions()

        if not distributions:
            return uncertainty

        # Monte Carlo simulation
        impact_samples = {cat: [] for cat in self.IMPACT_CATEGORIES}

        for _ in range(n_iterations):
            # Sample parameters
            sampled_params = {}
            for param, dist in distributions.items():
                if dist["type"] == "triangular":
                    sampled_params[param] = np.random.triangular(
                        dist["min"], dist["mode"], dist["max"]
                    )
                elif dist["type"] == "normal":
                    sampled_params[param] = np.random.normal(dist["mean"], dist["std"])
                elif dist["type"] == "uniform":
                    sampled_params[param] = np.random.uniform(dist["min"], dist["max"])

            # Calculate impacts with sampled parameters
            inventory = pathway.get_scaled_inventory(
                functional_unit_value * 1000, parameters=sampled_params
            )
            impacts = self.impact_assessment.calculate(inventory)

            for cat, value in impacts.items():
                impact_samples[cat].append(value)

        # Calculate statistics
        for cat, samples in impact_samples.items():
            samples = np.array(samples)
            uncertainty[cat] = {
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "cv": float(np.std(samples) / np.mean(samples)) if np.mean(samples) != 0 else 0,
                "p5": float(np.percentile(samples, 5)),
                "p25": float(np.percentile(samples, 25)),
                "p50": float(np.percentile(samples, 50)),
                "p75": float(np.percentile(samples, 75)),
                "p95": float(np.percentile(samples, 95)),
            }

        return uncertainty

    def sensitivity_analysis(
        self,
        pathway,
        parameters: List[str],
        variation: float = 0.1,
        functional_unit_value: float = 1.0,
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform sensitivity analysis on specified parameters.

        Args:
            pathway: Treatment pathway
            parameters: Parameters to analyze
            variation: Variation fraction (e.g., 0.1 = ±10%)
            functional_unit_value: Functional unit amount

        Returns:
            Sensitivity coefficients for each parameter-impact pair
        """
        base_result = self.calculate(pathway, functional_unit_value, include_uncertainty=False)
        base_impacts = base_result.impacts

        sensitivity = {}

        for param in parameters:
            sensitivity[param] = {}

            # Calculate with +variation
            high_result = self._calculate_with_parameter_change(
                pathway, param, 1 + variation, functional_unit_value
            )

            # Calculate with -variation
            low_result = self._calculate_with_parameter_change(
                pathway, param, 1 - variation, functional_unit_value
            )

            # Calculate sensitivity coefficient for each impact
            for cat in self.IMPACT_CATEGORIES:
                if base_impacts.get(cat, 0) != 0:
                    delta_impact = high_result.impacts.get(cat, 0) - low_result.impacts.get(cat, 0)
                    sensitivity[param][cat] = delta_impact / (2 * variation * base_impacts[cat])
                else:
                    sensitivity[param][cat] = 0

        return sensitivity

    def _calculate_with_parameter_change(
        self, pathway, parameter: str, factor: float, functional_unit_value: float
    ) -> LCAResult:
        """Calculate LCA with a parameter modified by a factor."""
        # Create a copy of pathway with modified parameter
        modified_pathway = pathway.copy_with_modified_parameter(parameter, factor)
        return self.calculate(modified_pathway, functional_unit_value, include_uncertainty=False)

    def get_contribution_analysis(
        self, result: LCAResult, impact_category: str
    ) -> Dict[str, float]:
        """
        Analyze contribution of inventory items to an impact category.

        Args:
            result: LCA result
            impact_category: Impact category to analyze

        Returns:
            Dict mapping inventory item to contribution percentage
        """
        contributions = {}
        total_impact = result.impacts.get(impact_category, 0)

        if total_impact == 0:
            return contributions

        # Calculate contribution from each emission
        inventory = result.inventory
        cf = self.characterization.get_factors(impact_category)

        for emission_type in ["emissions_air", "emissions_water", "emissions_soil"]:
            for emission in inventory.get(emission_type, []):
                name = emission.get("name", "unknown")
                quantity = emission.get("quantity", 0)
                factor = cf.get(name, 0)

                contribution = quantity * factor
                if contribution != 0:
                    contributions[f"{emission_type}:{name}"] = (contribution / total_impact) * 100

        # Sort by absolute contribution
        contributions = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))

        return contributions


def main():
    # Example usage
    LCAEngine()
    # pathway = StackDisposalPathway(country="China")
    # engine = LCAEngine()
    # result = engine.calculate(pathway)
    # print(f"Climate Change: {result.impacts['climate_change']} kg CO2-eq")


if __name__ == "__main__":
    main()
