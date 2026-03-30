"""Soil Amendment Pathway (PG-SA)"""

from typing import Dict, List

from pgloop.lca.inventory import LifeCycleInventory
from pgloop.pathways.base_pathway import BasePathway


class SoilAmendmentPathway(BasePathway):
    """PG applied directly to agricultural soil."""

    @property
    def code(self) -> str:
        return "PG-Soil"

    @property
    def name(self) -> str:
        return "Soil Amendment"

    @property
    def trl(self) -> int:
        return 8

    def _default_parameters(self) -> Dict[str, float]:
        return {
            "transport_km": 30,
            "application_energy_kwh_per_t": 5,
            "ca_available_fraction": 0.23,
            "s_available_fraction": 0.18,
            "avoided_lime_factor": 0.5,
        }

    def _build_inventory(self) -> LifeCycleInventory:
        p = self.parameters
        lci = LifeCycleInventory("PG Soil Amendment", "1 kg PG", 1.0)
        lci.add_input("Phosphogypsum", 1.0, "kg")
        lci.add_input("Diesel", p["transport_km"] * 0.00003, "kg")
        lci.add_emission("CO2", 0.005, "kg", "air")
        lci.add_avoided_product("Agricultural lime", p["avoided_lime_factor"], "kg")
        return lci

    def get_capex_data(self) -> Dict:
        return {"equipment": [{"name": "Spreader", "cost": 50000}]}

    def get_opex_data(self) -> Dict:
        return {"labor": {"hours_per_tonne": 0.1}}

    def get_products(self) -> List[Dict]:
        return [{"name": "Soil conditioner", "quantity": 1.0, "unit": "kg", "price": 0.02}]
