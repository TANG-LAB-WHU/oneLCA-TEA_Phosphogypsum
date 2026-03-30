"""
TEA Engine Module

Main calculation engine for Techno-Economic Analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pgloop.tea.capex import CAPEXCalculator
from pgloop.tea.external_cost import ExternalCostCalculator
from pgloop.tea.opex import OPEXCalculator


@dataclass
class TEAResult:
    """Complete TEA result for a treatment pathway."""

    pathway_code: str
    functional_unit: str
    functional_unit_value: float

    # Conventional Life Cycle Cost
    capex_total: float  # Total capital expenditure
    capex_annualized: float  # Annualized CAPEX
    opex_total: float  # Total operational cost per FU
    revenue: float  # Revenue from products per FU
    clcc: float  # Conventional LCC per FU

    # Societal Life Cycle Cost
    internal_cost_shadow: float  # Internal cost at shadow prices
    external_cost: float  # External cost from emissions
    slcc: float  # Societal LCC per FU

    # Breakdown
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    revenue_breakdown: Dict[str, float] = field(default_factory=dict)
    external_breakdown: Dict[str, float] = field(default_factory=dict)

    # Uncertainty
    uncertainty: Dict[str, Dict] = field(default_factory=dict)

    metadata: Dict = field(default_factory=dict)


class TEAEngine:
    """
    Techno-Economic Analysis calculation engine.

    Methodologies:
    - CLCC: Conventional Life Cycle Costing
    - SLCC: Societal Life Cycle Costing (includes external costs)

    Functional Unit: 1 tonne phosphogypsum treated
    """

    FUNCTIONAL_UNIT = "1 tonne phosphogypsum treated"
    FUNCTIONAL_UNIT_VALUE = 1000  # kg

    # Default economic parameters
    DEFAULT_PARAMS = {
        "discount_rate": 0.05,  # 5%
        "lifetime_years": 20,
        "annual_operating_hours": 8000,
        "currency": "USD",
        "base_year": 2024,
    }

    def __init__(self, config_path: Optional[Path] = None, country: str = "global"):
        """
        Initialize the TEA engine.

        Args:
            config_path: Path to configuration files
            country: Target country for cost data
        """
        self.config_path = config_path or Path("./config")
        self.country = country
        self.params = self.DEFAULT_PARAMS.copy()

        self.capex_calc = CAPEXCalculator(self.params)
        self.opex_calc = OPEXCalculator(self.params, country)
        self.external_calc = ExternalCostCalculator(self.config_path)

        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from files."""
        # Load shadow prices if available
        shadow_file = self.config_path / "shadow_prices.yaml"
        if shadow_file.exists():
            try:
                import yaml

                with open(shadow_file, "r") as f:
                    prices = yaml.safe_load(f)
                    self.external_calc.update_prices(prices)
            except Exception:
                pass

    def calculate(
        self,
        pathway,  # Treatment pathway instance
        functional_unit_value: float = 1.0,  # tonnes
        include_external: bool = True,
        include_uncertainty: bool = False,
    ) -> TEAResult:
        """
        Calculate TEA for a treatment pathway.

        Args:
            pathway: Treatment pathway instance
            functional_unit_value: Amount in tonnes
            include_external: Include external costs
            include_uncertainty: Calculate uncertainty

        Returns:
            TEAResult with complete analysis
        """
        fu_kg = functional_unit_value * 1000  # Convert to kg

        # Get pathway data
        capex_data = pathway.get_capex_data()
        opex_data = pathway.get_opex_data()
        products = pathway.get_products()
        emissions = pathway.get_emissions() if include_external else {}

        # Calculate CAPEX
        capex_total = self.capex_calc.calculate_total(capex_data)
        annual_throughput = pathway.get_annual_throughput()  # tonnes/year
        capex_annualized = self.capex_calc.annualize(
            capex_total, self.params["discount_rate"], self.params["lifetime_years"]
        )
        capex_per_fu = capex_annualized / annual_throughput if annual_throughput > 0 else 0

        # Calculate OPEX
        opex_result = self.opex_calc.calculate(opex_data, fu_kg)
        opex_per_fu = opex_result["total"]

        # Calculate revenue
        revenue_result = self._calculate_revenue(products, fu_kg)
        revenue_per_fu = revenue_result["total"]

        # Conventional LCC
        clcc = capex_per_fu + opex_per_fu - revenue_per_fu

        # Societal LCC (if external costs included)
        internal_shadow = clcc * 1.0  # Apply transfer factor if available
        external_cost = 0.0
        external_breakdown = {}

        if include_external and emissions:
            external_result = self.external_calc.calculate(emissions)
            external_cost = external_result["total"]
            external_breakdown = external_result["breakdown"]

        slcc = internal_shadow + external_cost

        # Uncertainty (simplified)
        uncertainty = {}
        if include_uncertainty:
            uncertainty = self._calculate_uncertainty(pathway, functional_unit_value)

        return TEAResult(
            pathway_code=pathway.code,
            functional_unit=self.FUNCTIONAL_UNIT,
            functional_unit_value=functional_unit_value,
            capex_total=capex_total,
            capex_annualized=capex_annualized,
            opex_total=opex_per_fu,
            revenue=revenue_per_fu,
            clcc=clcc,
            internal_cost_shadow=internal_shadow,
            external_cost=external_cost,
            slcc=slcc,
            cost_breakdown={"capex_per_fu": capex_per_fu, **opex_result.get("breakdown", {})},
            revenue_breakdown=revenue_result.get("breakdown", {}),
            external_breakdown=external_breakdown,
            uncertainty=uncertainty,
            metadata={
                "country": self.country,
                "currency": self.params["currency"],
                "base_year": self.params["base_year"],
            },
        )

    def _calculate_revenue(self, products: List[Dict], functional_unit_kg: float) -> Dict:
        """Calculate revenue from products."""
        total = 0.0
        breakdown = {}

        for product in products:
            name = product.get("name", "unknown")
            quantity = product.get("quantity", 0) * (functional_unit_kg / 1000)  # Scale to FU
            price = product.get("price", 0)

            revenue = quantity * price
            total += revenue
            breakdown[name] = revenue

        return {"total": total, "breakdown": breakdown}

    def _calculate_uncertainty(
        self, pathway, functional_unit_value: float, n_iterations: int = 500
    ) -> Dict:
        """Monte Carlo uncertainty for costs."""
        clcc_samples = []
        slcc_samples = []

        distributions = pathway.get_cost_distributions()

        for _ in range(n_iterations):
            # Sample parameters
            sampled = {}
            for param, dist in distributions.items():
                if dist["type"] == "triangular":
                    sampled[param] = np.random.triangular(dist["min"], dist["mode"], dist["max"])
                elif dist["type"] == "uniform":
                    sampled[param] = np.random.uniform(dist["min"], dist["max"])

            # Calculate with sampled parameters
            try:
                result = self.calculate(
                    pathway.copy_with_parameters(sampled),
                    functional_unit_value,
                    include_external=True,
                    include_uncertainty=False,
                )
                clcc_samples.append(result.clcc)
                slcc_samples.append(result.slcc)
            except Exception:
                pass

        if not clcc_samples:
            return {}

        return {
            "clcc": {
                "mean": float(np.mean(clcc_samples)),
                "std": float(np.std(clcc_samples)),
                "p5": float(np.percentile(clcc_samples, 5)),
                "p95": float(np.percentile(clcc_samples, 95)),
            },
            "slcc": {
                "mean": float(np.mean(slcc_samples)),
                "std": float(np.std(slcc_samples)),
                "p5": float(np.percentile(slcc_samples, 5)),
                "p95": float(np.percentile(slcc_samples, 95)),
            },
        }

    def compare_pathways(
        self, pathways: List, functional_unit_value: float = 1.0
    ) -> Dict[str, TEAResult]:
        """Compare multiple treatment pathways."""
        results = {}
        for pathway in pathways:
            results[pathway.code] = self.calculate(pathway, functional_unit_value)
        return results

    def calculate_npv(
        self, pathway, project_lifetime: int = 20, annual_throughput: float = None
    ) -> Dict:
        """
        Calculate Net Present Value for a pathway.

        Args:
            pathway: Treatment pathway
            project_lifetime: Project lifetime in years
            annual_throughput: Tonnes processed per year

        Returns:
            Dict with NPV, IRR, payback period
        """
        throughput = annual_throughput or pathway.get_annual_throughput()

        # Get costs
        capex_data = pathway.get_capex_data()
        capex_total = self.capex_calc.calculate_total(capex_data)

        result = self.calculate(pathway, 1.0)
        annual_profit = (result.revenue - result.opex_total) * throughput

        # Cash flows
        cash_flows = [-capex_total]
        for year in range(1, project_lifetime + 1):
            cash_flows.append(annual_profit)

        # NPV
        discount_rate = self.params["discount_rate"]
        npv = sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(cash_flows))

        # Simple payback
        cumulative = -capex_total
        payback = None
        for year in range(1, project_lifetime + 1):
            cumulative += annual_profit
            if cumulative >= 0 and payback is None:
                payback = year

        return {
            "npv": npv,
            "payback_years": payback if payback is not None else project_lifetime,
            "annual_profit": annual_profit,
            "total_investment": capex_total,
        }


def main():
    # Example usage
    TEAEngine(country="China")
    # result = engine.calculate(pathway)
    # engine = TEAEngine(country="China")
    # print(f"CLCC: ${result.clcc:.2f}/tonne")
    # print(f"SLCC: ${result.slcc:.2f}/tonne")


if __name__ == "__main__":
    main()
