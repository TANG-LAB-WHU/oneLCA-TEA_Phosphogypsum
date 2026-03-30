"""
Scenario Analysis Module

Compare pathways under different scenarios with regional context.
Includes geography, resource endowments, and infrastructure factors.
"""

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class RegionalContext:
    """
    Regional context for scenario analysis.

    Captures geography, resource endowments, and infrastructure.
    """

    # Basic info
    region_name: str
    country: str

    # Geographic conditions
    climate: str = "temperate"  # arid/tropical/temperate/cold
    coastal_access: bool = True
    seismic_zone: int = 1  # 1-5 (1=low risk)

    # Resource endowments
    pg_availability_mt_yr: float = 1.0  # Million tonnes/year
    pg_quality: str = "medium"  # low/medium/high purity
    ree_content_ppm: float = 200.0  # REE concentration
    water_availability: str = "adequate"  # scarce/adequate/abundant
    energy_mix: Dict[str, float] = field(
        default_factory=lambda: {
            "coal": 0.30,
            "gas": 0.25,
            "nuclear": 0.15,
            "hydro": 0.15,
            "solar": 0.10,
            "wind": 0.05,
        }
    )

    # Infrastructure
    electricity_price_usd_kwh: float = 0.10
    natural_gas_price_usd_mmbtu: float = 5.0
    grid_emission_factor_kg_co2_kwh: float = 0.45
    port_distance_km: float = 50.0
    industrial_zone: bool = True

    # Labor & Market
    labor_cost_factor: float = 1.0  # Relative to US
    skilled_labor_availability: str = "adequate"
    local_market_demand: str = "medium"  # low/medium/high

    # Regulatory
    environmental_stringency: str = "moderate"  # lax/moderate/strict
    waste_disposal_regulations: str = "standard"

    def get_cost_multiplier(self) -> float:
        """Calculate regional cost multiplier."""
        multiplier = self.labor_cost_factor

        # Climate adjustment
        if self.climate == "arid":
            multiplier *= 1.05  # Cooling costs
        elif self.climate == "cold":
            multiplier *= 1.08  # Heating costs
        elif self.climate == "tropical":
            multiplier *= 1.03  # Humidity control

        # Infrastructure adjustment
        if self.port_distance_km > 100:
            multiplier *= 1 + (self.port_distance_km - 100) * 0.0005

        if not self.industrial_zone:
            multiplier *= 1.10

        return multiplier

    def get_gwp_adjustment(self) -> float:
        """Calculate GWP adjustment based on grid mix."""
        # Base: 0.45 kg CO2/kWh
        return self.grid_emission_factor_kg_co2_kwh / 0.45


# Predefined regional contexts
CHINA_YUNNAN = RegionalContext(
    region_name="Yunnan Province",
    country="China",
    climate="temperate",
    pg_availability_mt_yr=5.0,
    ree_content_ppm=350,
    water_availability="abundant",
    energy_mix={"hydro": 0.70, "coal": 0.15, "gas": 0.05, "solar": 0.05, "wind": 0.05},
    electricity_price_usd_kwh=0.06,
    grid_emission_factor_kg_co2_kwh=0.15,
    labor_cost_factor=0.55,
)

MOROCCO_JORF = RegionalContext(
    region_name="Jorf Lasfar",
    country="Morocco",
    climate="arid",
    coastal_access=True,
    pg_availability_mt_yr=30.0,  # OCP massive production
    ree_content_ppm=400,
    water_availability="scarce",
    electricity_price_usd_kwh=0.12,
    grid_emission_factor_kg_co2_kwh=0.55,
    labor_cost_factor=0.45,
    port_distance_km=5,
)

USA_FLORIDA = RegionalContext(
    region_name="Florida",
    country="USA",
    climate="tropical",
    coastal_access=True,
    pg_availability_mt_yr=20.0,
    ree_content_ppm=150,
    water_availability="adequate",
    electricity_price_usd_kwh=0.11,
    grid_emission_factor_kg_co2_kwh=0.40,
    labor_cost_factor=1.0,
    environmental_stringency="strict",
)

BRAZIL_MG = RegionalContext(
    region_name="Minas Gerais",
    country="Brazil",
    climate="tropical",
    pg_availability_mt_yr=8.0,
    ree_content_ppm=250,
    water_availability="adequate",
    energy_mix={"hydro": 0.65, "wind": 0.10, "solar": 0.05, "gas": 0.10, "biomass": 0.10},
    electricity_price_usd_kwh=0.08,
    grid_emission_factor_kg_co2_kwh=0.10,
    labor_cost_factor=0.50,
)


@dataclass
class Scenario:
    """
    A scenario with parameter modifications and regional context.
    """

    name: str
    description: str = ""
    parameters: Dict[str, float] = field(default_factory=dict)

    # Regional context
    context: Optional[RegionalContext] = None

    # Multipliers for quick adjustments
    price_multiplier: float = 1.0
    cost_multiplier: float = 1.0
    carbon_price_usd_t: float = 100.0
    subsidy_rate: float = 0.0

    # Time horizon
    year: int = 2024
    projection_years: int = 20

    def apply_to(self, base_params: Dict) -> Dict:
        """Apply scenario modifications to base parameters."""
        result = copy.deepcopy(base_params)

        # Apply explicit parameters
        for key, value in self.parameters.items():
            if key in result:
                result[key] = value

        # Apply regional context if available
        if self.context:
            result["electricity_price"] = self.context.electricity_price_usd_kwh
            result["labor_factor"] = self.context.labor_cost_factor
            result["grid_emission_factor"] = self.context.grid_emission_factor_kg_co2_kwh
            result["regional_cost_multiplier"] = self.context.get_cost_multiplier()

        # Apply multipliers
        effective_cost_mult = self.cost_multiplier
        if self.context:
            effective_cost_mult *= self.context.get_cost_multiplier()

        for key in result:
            if "price" in key.lower() or "revenue" in key.lower():
                if isinstance(result[key], (int, float)):
                    result[key] *= self.price_multiplier
            if "cost" in key.lower() or "capex" in key.lower() or "opex" in key.lower():
                if isinstance(result[key], (int, float)):
                    result[key] *= effective_cost_mult

        return result

    def with_context(self, context: RegionalContext) -> "Scenario":
        """Create new scenario with regional context."""
        new_scenario = copy.deepcopy(self)
        new_scenario.context = context
        new_scenario.name = f"{self.name}_{context.region_name}"
        return new_scenario


# Predefined market scenarios
BASELINE = Scenario(
    name="baseline",
    description="Current market conditions",
    carbon_price_usd_t=100,
)

OPTIMISTIC = Scenario(
    name="optimistic",
    description="Favorable market conditions",
    price_multiplier=1.2,
    cost_multiplier=0.9,
    carbon_price_usd_t=150,
    subsidy_rate=0.15,
)

PESSIMISTIC = Scenario(
    name="pessimistic",
    description="Adverse market conditions",
    price_multiplier=0.8,
    cost_multiplier=1.15,
    carbon_price_usd_t=50,
    subsidy_rate=0.0,
)

HIGH_CARBON_PRICE = Scenario(
    name="high_carbon_price",
    description="Strong carbon pricing regime",
    carbon_price_usd_t=250,
)

LOW_SUBSIDY = Scenario(
    name="low_subsidy",
    description="Reduced government support",
    subsidy_rate=0.0,
    cost_multiplier=1.05,
)

# Future projections
SCENARIO_2030 = Scenario(
    name="projection_2030",
    description="2030 market projections",
    year=2030,
    carbon_price_usd_t=150,
    price_multiplier=1.1,
    parameters={"ree_price_multiplier": 1.5},
)

SCENARIO_2040 = Scenario(
    name="projection_2040",
    description="2040 market projections",
    year=2040,
    carbon_price_usd_t=200,
    price_multiplier=1.2,
    parameters={"ree_price_multiplier": 2.0},
)

# Registry for easy access
MARKET_SCENARIOS = {
    "baseline": BASELINE,
    "optimistic": OPTIMISTIC,
    "pessimistic": PESSIMISTIC,
    "high_carbon": HIGH_CARBON_PRICE,
    "low_subsidy": LOW_SUBSIDY,
    "2030": SCENARIO_2030,
    "2040": SCENARIO_2040,
}


@dataclass
class ScenarioResult:
    """Result for a pathway under a scenario."""

    pathway_name: str
    scenario_name: str
    metrics: Dict[str, float]
    context: Optional[RegionalContext] = None


class ScenarioAnalyzer:
    """
    Analyzes pathways across multiple scenarios and regional contexts.
    """

    def __init__(
        self,
        lca_engine=None,
        tea_engine=None,
        scenarios: List[Scenario] = None,
        contexts: List[RegionalContext] = None,
    ):
        self.lca_engine = lca_engine
        self.tea_engine = tea_engine
        self.scenarios = scenarios or [BASELINE, OPTIMISTIC, PESSIMISTIC]
        self.contexts = contexts or [CHINA_YUNNAN, MOROCCO_JORF, USA_FLORIDA, BRAZIL_MG]

    def add_scenario(self, scenario: Scenario) -> None:
        self.scenarios.append(scenario)

    def add_context(self, context: RegionalContext) -> None:
        self.contexts.append(context)

    def analyze(
        self,
        pathways: Dict[str, Dict[str, float]],
        evaluation_fn: Callable = None,
        include_contexts: bool = True,
    ) -> Dict[str, List[ScenarioResult]]:
        """
        Analyze pathways under all scenarios and contexts.

        Args:
            pathways: {pathway_name: base_metrics}
            evaluation_fn: Optional function to recalculate metrics
            include_contexts: Whether to cross with regional contexts

        Returns:
            {pathway_name: [ScenarioResult for each scenario×context]}
        """
        results = {}

        # Build scenario list (with and without contexts)
        all_scenarios = list(self.scenarios)

        if include_contexts and self.contexts:
            for scenario in self.scenarios:
                for context in self.contexts:
                    all_scenarios.append(scenario.with_context(context))

        for pathway_name, base_metrics in pathways.items():
            pathway_results = []

            for scenario in all_scenarios:
                if evaluation_fn:
                    scenario_params = scenario.apply_to(base_metrics)
                    metrics = evaluation_fn(scenario_params)
                else:
                    metrics = self._apply_simple_adjustment(base_metrics, scenario)

                pathway_results.append(
                    ScenarioResult(
                        pathway_name=pathway_name,
                        scenario_name=scenario.name,
                        metrics=metrics,
                        context=scenario.context,
                    )
                )

            results[pathway_name] = pathway_results

        return results

    def _apply_simple_adjustment(
        self, metrics: Dict[str, float], scenario: Scenario
    ) -> Dict[str, float]:
        """Apply scenario adjustments to metrics."""
        result = copy.deepcopy(metrics)

        # Cost multiplier with regional adjustment
        cost_mult = scenario.cost_multiplier
        if scenario.context:
            cost_mult *= scenario.context.get_cost_multiplier()

        if "npv" in result:
            result["npv"] *= scenario.price_multiplier / cost_mult

        if "capex" in result:
            result["capex"] *= cost_mult

        if "opex" in result:
            result["opex"] *= cost_mult

        # GWP adjustment for grid mix
        if "gwp" in result and scenario.context:
            result["gwp"] *= scenario.context.get_gwp_adjustment()

        # Carbon cost
        if "gwp" in result:
            result["carbon_cost"] = result["gwp"] * scenario.carbon_price_usd_t / 1000

        return result

    def compare_robustness(
        self, analysis_results: Dict[str, List[ScenarioResult]], metric: str = "npv"
    ) -> Dict[str, Dict]:
        """Compare pathway robustness across scenarios."""
        robustness = {}

        for pathway_name, results in analysis_results.items():
            values = [r.metrics.get(metric, 0) for r in results]

            if values:
                mean = sum(values) / len(values)
                robustness[pathway_name] = {
                    "mean": mean,
                    "std": (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5,
                    "min": min(values),
                    "max": max(values),
                    "range": max(values) - min(values),
                    "cv": (
                        (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5 / abs(mean)
                        if mean
                        else 0
                    ),
                    "scenarios": {r.scenario_name: r.metrics.get(metric, 0) for r in results},
                }

        return robustness

    def quick_robustness_check(
        self, pathway, scenarios: List[Scenario] = None, metric: str = "npv"
    ) -> Dict[str, float]:
        """
        Quickly check a single pathway's performance across scenarios.
        """
        scens = scenarios or self.scenarios
        results = {}

        # We need engines for a 'real' check, but for demo we can mock if not present
        for s in scens:
            # If we have engines, we could do full recalc,
            # for now we'll use simple adjustment
            # In a real app, this would use lca_engine and tea_engine
            base_metrics = {
                "npv": 10.0,
                "clcc": 50.0,
                "gwp": 100.0,
                "trl": getattr(pathway, "trl", 9),
            }
            adj = self._apply_simple_adjustment(base_metrics, s)
            results[s.name] = adj.get(metric, 0)

        return results

    def best_region_for_pathway(
        self,
        analysis_results: Dict[str, List[ScenarioResult]],
        pathway_name: str,
        metric: str = "npv",
    ) -> Optional[str]:
        """Find best regional context for a pathway."""
        if pathway_name not in analysis_results:
            return None

        results_with_context = [r for r in analysis_results[pathway_name] if r.context is not None]

        if not results_with_context:
            return None

        best = max(results_with_context, key=lambda r: r.metrics.get(metric, 0))
        return best.context.region_name if best.context else None
