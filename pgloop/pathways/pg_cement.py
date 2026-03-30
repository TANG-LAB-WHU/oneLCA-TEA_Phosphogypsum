"""
Cement Production Pathway (PG-CM)

Phosphogypsum used as cement additive/retarder.
"""

from typing import Dict, List

from pgloop.lca.inventory import LifeCycleInventory
from pgloop.pathways.base_pathway import BasePathway


class CementPathway(BasePathway):
    """
    Cement Production Pathway.

    Phosphogypsum replaces natural gypsum in cement production:
    - 3-5% of cement composition
    - Acts as set retarder
    - Reduces natural gypsum mining
    """

    @property
    def code(self) -> str:
        return "PG-CementProd"

    @property
    def name(self) -> str:
        return "Cement Additive"

    @property
    def trl(self) -> int:
        return 9

    def _default_parameters(self) -> Dict[str, float]:
        return {
            # Processing
            "drying_energy_mj_per_t": 150,  # For moisture removal
            "electricity_kwh_per_t": 20,
            "transport_km": 50,
            # Substitution
            "gypsum_substitution_ratio": 0.95,  # kg natural gypsum avoided per kg PG
            # PG composition
            "caso4_fraction": 0.92,
            "moisture_fraction": 0.20,
            "ra226_bq_kg": 500,
            # Emissions
            "drying_co2_kg_per_mj": 0.055,  # Natural gas heating
            # Product quality
            "usable_fraction": 0.90,  # Fraction suitable for cement
        }

    def _build_inventory(self) -> LifeCycleInventory:
        p = self.parameters

        lci = LifeCycleInventory(
            process_name="PG for Cement Production",
            functional_unit="1 kg PG",
            functional_unit_value=1.0,
        )

        # Inputs
        lci.add_input("Phosphogypsum", 1.0, "kg")

        # Energy for drying
        drying_energy = p["drying_energy_mj_per_t"] / 1000
        lci.add_input("Natural gas", drying_energy, "MJ")
        lci.add_input("Electricity", p["electricity_kwh_per_t"] / 1000, "kWh")

        # Transport
        diesel_transport = p["transport_km"] * 0.00003  # kg diesel per kg-km
        lci.add_input("Diesel", diesel_transport, "kg")

        # Emissions
        co2_drying = drying_energy * p["drying_co2_kg_per_mj"]
        co2_transport = diesel_transport * 3.2
        lci.add_emission("CO2", co2_drying + co2_transport, "kg", "air")

        # Water vapor from drying
        water_released = p["moisture_fraction"]
        lci.add_emission("Water vapor", water_released, "kg", "air")

        # Avoided product: natural gypsum
        avoided_gypsum = p["usable_fraction"] * p["gypsum_substitution_ratio"]
        lci.add_avoided_product("Natural gypsum", avoided_gypsum, "kg")

        return lci

    def get_capex_data(self) -> Dict:
        return {
            "equipment": [
                {"name": "Rotary dryer", "cost": 800000},
                {"name": "Conveyor system", "cost": 200000},
                {"name": "Storage silos", "cost": 300000},
                {"name": "Quality control lab", "cost": 100000},
            ],
            "factors": {
                "installation": 1.3,
                "engineering": 0.12,
                "contingency": 0.15,
            },
        }

    def get_opex_data(self) -> Dict:
        p = self.parameters
        return {
            "materials": [],
            "utilities": {
                "electricity_kwh": p["electricity_kwh_per_t"],
                "heat_mj": p["drying_energy_mj_per_t"],
            },
            "labor": {
                "hours_per_tonne": 0.2,
            },
            "maintenance": 80000,
        }

    def get_products(self) -> List[Dict]:
        p = self.parameters
        return [
            {
                "name": "Cement-grade gypsum",
                "quantity": p["usable_fraction"],  # Per kg PG input
                "unit": "kg",
                "price": 0.03,  # USD/kg (gypsum market price)
            }
        ]


def main():
    pathway = CementPathway(country="China")
    print(f"Pathway: {pathway.name} ({pathway.code})")

    inv = pathway.get_scaled_inventory(1000)
    for product in inv.get("avoided_products", []):
        print(f"Avoided: {product}")


if __name__ == "__main__":
    main()
