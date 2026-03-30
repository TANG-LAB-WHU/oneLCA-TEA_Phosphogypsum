"""
REE Extraction Pathway (PG-RE)

Extraction of Rare Earth Elements (REE) from phosphogypsum via acid leaching.
"""

from typing import Dict, List

from pgloop.lca.inventory import LifeCycleInventory
from pgloop.pathways.base_pathway import BasePathway


class REEExtractionPathway(BasePathway):
    """
    REE recovery from phosphogypsum using hydrometallurgy.

    Typical process involving:
    1. Acid leaching (H2SO4 or HCl)
    2. Solvent extraction / Ion exchange
    3. Precipitation and purification
    """

    @property
    def code(self) -> str:
        return "PG-REEextract"

    @property
    def name(self) -> str:
        return "REE Extraction"

    @property
    def trl(self) -> int:
        return 5  # Pilot / Lab scale

    def _default_parameters(self) -> Dict[str, float]:
        """Detailed parameters for REE extraction."""
        return {
            # Feedstock
            "ree_content_ppm": 350.0,  # Mg/kg in dry PG
            "moisture_content": 0.20,  # 20% moisture
            # Process Efficiency
            "leaching_efficiency": 0.65,  # 65% yield from leaching
            "purification_efficiency": 0.90,  # 90% yield from SX
            # Materials Consumption (per tonne dry PG)
            "h2so4_kg_per_t": 250.0,  # Sulfuric acid
            "sodium_hydroxide_kg_per_t": 40.0,  # Precipitation
            "extractant_l_per_t": 0.5,  # Solvent extraction loss
            "process_water_m3_per_t": 5.0,
            # Energy (per tonne dry PG)
            "electricity_kwh_per_t": 120.0,
            "steam_mj_per_t": 450.0,  # Heating for leaching
            # Waste Management
            "residue_treatment_cost_usd_t": 5.0,
            "liquid_effluent_m3_per_t": 4.5,
            # Economics
            "ree_market_price_usd_kg": 45.0,  # Mixed REE carbonate/oxide
        }

    def _build_inventory(self) -> LifeCycleInventory:
        """Build detailed LCI for REE extraction."""
        p = self.parameters

        lci = LifeCycleInventory(
            process_name="PG REE Extraction (Hydrometallurgy)",
            functional_unit="1 kg dry PG",
            functional_unit_value=1.0,
        )

        # Inputs (all scaled to 1 kg dry PG)
        lci.add_input("Phosphogypsum", 1.0, "kg")
        lci.add_input("Sulfuric acid", p["h2so4_kg_per_t"] / 1000, "kg")
        lci.add_input("Sodium hydroxide", p["sodium_hydroxide_kg_per_t"] / 1000, "kg")
        lci.add_input("Electricity", p["electricity_kwh_per_t"] / 1000, "kWh")
        lci.add_input("Process water", p["process_water_m3_per_t"] / 1000, "m3")
        lci.add_input("Heat (Steam)", p["steam_mj_per_t"] / 1000, "MJ")

        # Output: REE Concentrate (kg per kg dry PG)
        # ppm / 1,000,000 * eff_leach * eff_purif
        ree_yield_kg_kg = (
            (p["ree_content_ppm"] / 1e6) * p["leaching_efficiency"] * p["purification_efficiency"]
        )
        lci.add_output("REE Concentrate", ree_yield_kg_kg, "kg")

        # Emissions
        # Fuel-based emissions for steam (assuming natural gas)
        lci.add_emission("CO2", (p["steam_mj_per_t"] / 1000) * 0.056, "kg", "air")

        # Aqueous emissions (fluoride and phosphate in effluent)
        lci.add_emission("Phosphate", 0.0005, "kg", "water")
        lci.add_emission("Fluoride", 0.0002, "kg", "water")

        return lci

    def get_capex_data(self) -> Dict:
        """Hydrometallurgical plant CAPEX estimates."""
        return {
            "equipment": [
                {"name": "Crushing & Milling", "cost": 1500000},
                {"name": "Leaching Reactors (Glass-lined)", "cost": 3000000},
                {"name": "Filtration Units", "cost": 1200000},
                {"name": "Solvent Extraction Battery", "cost": 4500000},
                {"name": "Precipitation & Drying", "cost": 1800000},
                {"name": "Waste Water Treatment", "cost": 2500000},
            ],
            "factors": {
                "piping_instrumentation": 0.45,
                "engineering": 0.15,
                "contingency": 0.20,
            },
        }

    def get_opex_data(self) -> Dict:
        """OPEX per tonne processed."""
        p = self.parameters
        return {
            "materials": [
                {"name": "H2SO4", "quantity": p["h2so4_kg_per_t"], "price": 0.09, "unit": "kg"},
                {
                    "name": "NaOH",
                    "quantity": p["sodium_hydroxide_kg_per_t"],
                    "price": 0.45,
                    "unit": "kg",
                },
                {
                    "name": "Extractants",
                    "quantity": p["extractant_l_per_t"],
                    "price": 12.0,
                    "unit": "L",
                },
            ],
            "utilities": {
                "electricity_kwh": p["electricity_kwh_per_t"],
                "heat_mj": p["steam_mj_per_t"],
                "water_m3": p["process_water_m3_per_t"],
            },
            "labor": {
                "hours_per_tonne": 1.5,  # More labor intensive than stack disposal
            },
            "waste": {
                "residual_pg_disposal": p["residue_treatment_cost_usd_t"],
            },
            "maintenance_rate": 0.04,  # 4% of CAPEX
        }

    def get_products(self) -> List[Dict]:
        """Mixed REE products."""
        p = self.parameters
        ree_yield_kg_t = (
            (p["ree_content_ppm"] / 1000) * p["leaching_efficiency"] * p["purification_efficiency"]
        )
        return [
            {
                "name": "Mixed REE Concentrate",
                "quantity": ree_yield_kg_t / 1000,  # kg REE per kg dry PG
                "unit": "kg",
                "price": p["ree_market_price_usd_kg"],
            }
        ]


def main():
    pathway = REEExtractionPathway(country="Morocco")
    print(f"Pathway: {pathway.name}")
    print(f"Functional Unit: {pathway.get_scaled_inventory(1.0)['functional_unit']}")

    prods = pathway.get_products()
    print(f"REE Yield: {prods[0]['quantity'] * 1000:.4f} kg/t dry PG")
    print(f"Product Value: ${prods[0]['quantity'] * 1000 * prods[0]['price']:.2f}/t")


if __name__ == "__main__":
    main()
