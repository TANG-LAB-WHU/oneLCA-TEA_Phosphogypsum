"""
Sulfur & Sulfuric Acid Recovery Pathway (PG-SulfurAcid)

Phosphogypsum thermochemical decomposition to produce sulfur, sulfuric acid, and calcium oxide residue.
"""

from typing import Dict, List

from pgloop.lca.inventory import LifeCycleInventory
from pgloop.pathways.base_pathway import BasePathway


class SulfurAcidPathway(BasePathway):
    """
    Sulfur & Sulfuric Acid Recovery Pathway.

    Phosphogypsum is decomposed at high temperatures under reducing conditions:
    - CaSO4 + CO/C -> CaO + SO2 + CO2
    - SO2 is processed to produce Sulfuric Acid (H2SO4)
    - Under optimized reducing conditions, elemental Sulfur (S) is recovered
    - Calcium oxide (CaO) residue can be used as cement clinker component
    """

    @property
    def code(self) -> str:
        return "PG-SulfurAcid"

    @property
    def name(self) -> str:
        return "Sulfur & Acid Recovery"

    @property
    def trl(self) -> int:
        return 7

    def _default_parameters(self) -> Dict[str, float]:
        return {
            # Processing inputs (per tonne PG input)
            "coal_reducing_kg_per_t": 80.0,
            "coal_heating_kg_per_t": 100.0,
            "additives_kg_per_t": 180.0,
            "electricity_kwh_per_t": 80.0,
            # Product yields (kg product per kg PG input)
            "sulfuric_acid_yield": 0.55,
            "sulfur_yield": 0.10,
            "clinker_yield": 0.35,
            # Emissions
            "co2_emissions_kg_per_t": 350.0,
            "moisture_fraction": 0.20,
        }

    def _build_inventory(self) -> LifeCycleInventory:
        p = self.parameters

        lci = LifeCycleInventory(
            process_name="PG Decomposition to Sulfur & Acid",
            functional_unit="1 kg PG",
            functional_unit_value=1.0,
        )

        # Inputs
        lci.add_input("Phosphogypsum", 1.0, "kg")

        coal_input = (p["coal_reducing_kg_per_t"] + p["coal_heating_kg_per_t"]) / 1000.0
        lci.add_input("Coal", coal_input, "kg")
        lci.add_input("Silica-Alumina Additives", p["additives_kg_per_t"] / 1000.0, "kg")
        lci.add_input("Electricity", p["electricity_kwh_per_t"] / 1000.0, "kWh")

        # Outputs
        lci.add_output("Sulfuric acid", p["sulfuric_acid_yield"], "kg")
        lci.add_output("Sulfur", p["sulfur_yield"], "kg")
        lci.add_output("Calcium oxide residue", p["clinker_yield"], "kg")

        # Emissions
        lci.add_emission("CO2", p["co2_emissions_kg_per_t"] / 1000.0, "kg", "air")
        lci.add_emission("Water vapor", p["moisture_fraction"], "kg", "air")

        return lci

    def get_capex_data(self) -> Dict:
        return {
            "equipment": [
                {"name": "Rotary kiln system", "cost": 6500000},
                {"name": "Acid absorption tower", "cost": 2200000},
                {"name": "Sulfur recovery unit", "cost": 1500000},
                {"name": "Gas purification system", "cost": 1200000},
                {"name": "Material dosing & mixing system", "cost": 800000},
            ],
            "factors": {
                "installation": 1.4,
                "engineering": 0.15,
                "contingency": 0.18,
            },
        }

    def get_opex_data(self) -> Dict:
        p = self.parameters
        return {
            "materials": [
                {
                    "name": "Coal",
                    "quantity": p["coal_reducing_kg_per_t"] + p["coal_heating_kg_per_t"],
                    "per_kg_input": 1000,
                    "price": 0.12,
                },
                {
                    "name": "Silica-Alumina Additives",
                    "quantity": p["additives_kg_per_t"],
                    "per_kg_input": 1000,
                    "price": 0.02,
                },
            ],
            "utilities": {
                "electricity_kwh": p["electricity_kwh_per_t"],
            },
            "labor": {
                "hours_per_tonne": 0.35,
            },
            "maintenance": 250000,
        }

    def get_products(self) -> List[Dict]:
        p = self.parameters
        return [
            {
                "name": "Sulfuric acid",
                "quantity": p["sulfuric_acid_yield"],
                "unit": "kg",
                "price": 0.10,
            },
            {
                "name": "Sulfur",
                "quantity": p["sulfur_yield"],
                "unit": "kg",
                "price": 0.12,
            },
            {
                "name": "Calcium oxide residue",
                "quantity": p["clinker_yield"],
                "unit": "kg",
                "price": 0.04,
            },
        ]


def main():
    pathway = SulfurAcidPathway(country="China")
    print(f"Pathway: {pathway.name} ({pathway.code})")

    inv = pathway.get_scaled_inventory(1000)
    for product in inv.get("outputs", []):
        print(f"Product: {product}")


if __name__ == "__main__":
    main()
