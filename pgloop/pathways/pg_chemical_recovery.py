"""Chemical Recovery Pathway (PG-CR)"""

from typing import Dict, List
from pgloop.pathways.base_pathway import BasePathway
from pgloop.lca.inventory import LifeCycleInventory


class ChemicalRecoveryPathway(BasePathway):
    """PG converted to ammonium sulfate and calcium carbonate."""
    
    @property
    def code(self) -> str: return "PG-ChemReco"
    
    @property
    def name(self) -> str: return "Chemical Recovery"
    
    @property
    def trl(self) -> int: return 7
    
    def _default_parameters(self) -> Dict[str, float]:
        return {
            "ammonia_kg_per_t": 200,
            "co2_input_kg_per_t": 500,
            "energy_kwh_per_t": 100,
            "ammonium_sulfate_yield": 0.8,
            "caco3_yield": 0.6,
        }
    
    def _build_inventory(self) -> LifeCycleInventory:
        p = self.parameters
        lci = LifeCycleInventory("PG Chemical Recovery", "1 kg PG", 1.0)
        lci.add_input("Phosphogypsum", 1.0, "kg")
        lci.add_input("Ammonia", p["ammonia_kg_per_t"] / 1000, "kg")
        lci.add_input("CO2 (captured)", p["co2_input_kg_per_t"] / 1000, "kg")
        lci.add_input("Electricity", p["energy_kwh_per_t"] / 1000, "kWh")
        lci.add_output("Ammonium sulfate", p["ammonium_sulfate_yield"], "kg")
        lci.add_output("Calcium carbonate", p["caco3_yield"], "kg")
        return lci
    
    def get_capex_data(self) -> Dict:
        return {"equipment": [{"name": "Reactor system", "cost": 5000000}]}
    
    def get_opex_data(self) -> Dict:
        return {
            "materials": [{"name": "Ammonia", "quantity": 200, "per_kg_input": 1000, "price": 0.40}],
            "utilities": {"electricity_kwh": 100},
        }
    
    def get_products(self) -> List[Dict]:
        p = self.parameters
        return [
            {"name": "Ammonium sulfate", "quantity": p["ammonium_sulfate_yield"], "unit": "kg", "price": 0.20},
            {"name": "Calcium carbonate", "quantity": p["caco3_yield"], "unit": "kg", "price": 0.08},
        ]
