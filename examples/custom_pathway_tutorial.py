"""
Custom Pathway Tutorial for PhosphogypsumBot.

This script demonstrates how to define a new custom PG treatment pathway
by subclassing BasePathway, specifying materials inventory, CAPEX/OPEX,
and running LCA and TEA engines on the custom pathway.
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pgloop import LCAEngine, TEAEngine
from pgloop.lca.inventory import LifeCycleInventory
from pgloop.pathways.base_pathway import BasePathway
from pgloop.utils.currency import format_currency


# 1. Define the Custom Pathway Class
class CarbonMineralizationPathway(BasePathway):
    """
    Custom Pathway: Carbon Mineralization using Slag and PG.
    
    This process sequestrates CO2 into carbonated building blocks.
    - Replaces natural aggregates.
    - Captures CO2 permanently.
    """

    @property
    def code(self) -> str:
        return "PG-CarbonMin"

    @property
    def name(self) -> str:
        return "CO2 Carbon Mineralization"

    @property
    def trl(self) -> int:
        return 6  # Technology Readiness Level 6 (Pilot scale)

    def _default_parameters(self) -> Dict[str, float]:
        """Define default parameters per 1 tonne PG input."""
        return {
            "slag_kg_per_t": 150.0,
            "co2_captured_kg_per_t": 220.0,
            "electricity_kwh_per_t": 75.0,
            "additive_kg_per_t": 30.0,
            "water_kg_per_t": 250.0,
            "product_yield": 1.15,          # 1.15 t block per t PG
            "avoided_aggregate_kg": 0.90,    # substitutes 0.90 kg aggregate per kg PG
            "process_co2_emission_kg": 25.0, # process emission
        }

    def _build_inventory(self) -> LifeCycleInventory:
        """Construct the Life Cycle Inventory (LCI) for LCA calculations."""
        p = self.parameters
        lci = LifeCycleInventory(
            process_name="PG Carbon Mineralization Process",
            functional_unit="1 kg PG",
            functional_unit_value=1.0
        )

        # Inputs
        lci.add_input("Phosphogypsum", 1.0, "kg")
        lci.add_input("Blast furnace slag", p["slag_kg_per_t"] / 1000.0, "kg")
        lci.add_input("Silica additive", p["additive_kg_per_t"] / 1000.0, "kg")
        lci.add_input("Electricity", p["electricity_kwh_per_t"] / 1000.0, "kWh")
        lci.add_input("Water", p["water_kg_per_t"] / 1000.0, "kg")

        # Carbon sequestration input (negative emission)
        lci.add_input("CO2 (captured)", p["co2_captured_kg_per_t"] / 1000.0, "kg")

        # Outputs
        lci.add_output("Mineralized blocks", p["product_yield"], "kg")

        # Emissions
        lci.add_emission("CO2", p["process_co2_emission_kg"] / 1000.0, "kg", "air")

        # Avoided Products (Credits)
        avoided = p["product_yield"] * p["avoided_aggregate_kg"]
        lci.add_avoided_product("Natural aggregate", avoided, "kg")

        return lci

    def get_capex_data(self) -> Dict:
        """Define equipment cost lists for Capital Expenditure (CAPEX) calculations."""
        return {
            "equipment": [
                {"name": "Carbonation reactor", "cost": 3200000},
                {"name": "Slurry mixer", "cost": 650000},
                {"name": "CO2 compression system", "cost": 1200000},
                {"name": "Block press machine", "cost": 850000},
            ],
            "factors": {
                "installation": 1.35,
                "engineering": 0.12,
                "contingency": 0.15,
            }
        }

    def get_opex_data(self) -> Dict:
        """Define variable raw materials, utility values and labor for OPEX."""
        p = self.parameters
        return {
            "materials": [
                {"name": "Blast furnace slag", "quantity": p["slag_kg_per_t"], "per_kg_input": 1000.0, "price": 0.04},
                {"name": "Silica additive", "quantity": p["additive_kg_per_t"], "per_kg_input": 1000.0, "price": 0.15},
                {"name": "CO2 purchase", "quantity": p["co2_captured_kg_per_t"], "per_kg_input": 1000.0, "price": 0.05},
            ],
            "utilities": {
                "electricity_kwh": p["electricity_kwh_per_t"],
                "water_m3": p["water_kg_per_t"] / 1000.0,
            },
            "labor": {
                "hours_per_tonne": 0.45,
            }
        }

    def get_products(self) -> List[Dict]:
        """Define co-products sales prices."""
        p = self.parameters
        return [
            {
                "name": "Mineralized blocks",
                "quantity": p["product_yield"],
                "unit": "kg",
                "price": 0.06,  # $60/t blocks
            }
        ]


# 2. Main execution script
def main():
    print("=" * 60)
    print("   PhosphogypsumBot: Custom Pathway Creation Tutorial   ")
    print("=" * 60)

    # A. Instantiate the custom pathway
    custom_pathway = CarbonMineralizationPathway()
    print(f"\nCreated pathway: {custom_pathway.name} [{custom_pathway.code}]")
    print(f"TRL Level: {custom_pathway.trl}")

    # B. Initialize evaluation engines
    lca_engine = LCAEngine()
    tea_engine = TEAEngine(country="China")

    # C. Calculate LCA
    print("\n[LCA] Evaluating environmental indices...")
    lca_result = lca_engine.calculate(custom_pathway, functional_unit_value=1.0)
    
    # Notice that climate change (GWP) will be negative due to carbon capture
    gwp = lca_result.impacts.get("climate_change", 0.0)
    print(f"  - Climate Change (GWP)     : {gwp:.3f} kg CO2-eq/t PG")
    print(f"  - Resource Depletion       : {lca_result.impacts.get('resource_depletion', 0.0):.6f} kg Sb-eq/t PG")

    # D. Calculate TEA
    print("\n[TEA] Evaluating financial indices...")
    tea_result = tea_engine.calculate(custom_pathway, functional_unit_value=1.0)
    npv_result = tea_engine.calculate_npv(custom_pathway)

    print(f"  - Conventional Cost (CLCC) : {format_currency(tea_result.clcc)} / t PG")
    print(f"  - Societal Cost (SLCC)     : {format_currency(tea_result.slcc)} / t PG (including carbon credit)")
    print(f"  - Net Present Value (NPV)  : {format_currency(npv_result['npv'])}")
    print(f"  - Payback Years            : {npv_result['payback_years']:.1f} years")

    print("\n" + "=" * 60)
    print("Tutorial Successfully Completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
