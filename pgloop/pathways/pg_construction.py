"""Construction Materials Pathway (PG-CB)"""

from typing import Dict, List
from pgloop.pathways.base_pathway import BasePathway
from pgloop.lca.inventory import LifeCycleInventory


class ConstructionMaterialsPathway(BasePathway):
    """PG used in bricks, plasterboard, road base."""
    
    @property
    def code(self) -> str: return "PG-ConstructMat"
    
    @property
    def name(self) -> str: return "Construction Materials"
    
    @property
    def trl(self) -> int: return 8
    
    def _default_parameters(self) -> Dict[str, float]:
        return {
            "processing_energy_kwh_per_t": 50,
            "binder_kg_per_t": 50,
            "water_kg_per_t": 200,
            "product_yield": 0.95,
            "avoided_aggregate_kg": 0.8,
        }
    
    def _build_inventory(self) -> LifeCycleInventory:
        p = self.parameters
        lci = LifeCycleInventory("PG Construction Materials", "1 kg PG", 1.0)
        lci.add_input("Phosphogypsum", 1.0, "kg")
        lci.add_input("Electricity", p["processing_energy_kwh_per_t"] / 1000, "kWh")
        lci.add_input("Cement binder", p["binder_kg_per_t"] / 1000, "kg")
        lci.add_input("Water", p["water_kg_per_t"] / 1000, "kg")
        lci.add_emission("CO2", 0.02, "kg", "air")  # Processing
        lci.add_avoided_product("Natural aggregate", p["avoided_aggregate_kg"], "kg")
        return lci
    
    def get_capex_data(self) -> Dict:
        return {"equipment": [{"name": "Block plant", "cost": 1500000}]}
    
    def get_opex_data(self) -> Dict:
        return {"utilities": {"electricity_kwh": 50}, "labor": {"hours_per_tonne": 0.3}}
    
    def get_products(self) -> List[Dict]:
        return [{"name": "Construction blocks", "quantity": 0.95, "unit": "kg", "price": 0.05}]
