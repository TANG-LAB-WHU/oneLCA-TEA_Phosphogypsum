"""
Stack Disposal Pathway (PG-SD)

Baseline: Phosphogypsum disposed in engineered stacks.
"""

from typing import Dict, List
from pgloop.pathways.base_pathway import BasePathway
from pgloop.lca.inventory import LifeCycleInventory


class StackDisposalPathway(BasePathway):
    """
    Stack Disposal Pathway (Baseline).
    
    Phosphogypsum is disposed in engineered stacks with:
    - Liner systems
    - Leachate collection
    - Radon monitoring
    - Long-term maintenance
    """
    
    @property
    def code(self) -> str:
        return "PG-Stack"
    
    @property
    def name(self) -> str:
        return "Stack Disposal"
    
    @property
    def trl(self) -> int:
        return 9
    
    def _default_parameters(self) -> Dict[str, float]:
        """Default parameters for stack disposal."""
        return {
            # Energy consumption
            "diesel_kg_per_t": 2.0,  # Transport and handling
            "electricity_kwh_per_t": 5.0,  # Pumping, monitoring
            
            # Emissions factors
            "co2_factor": 3.2,  # kg CO2 per kg diesel
            "radon_emission_rate": 0.5,  # Bq/m2/s average
            "stack_area_m2_per_t": 2.0,  # Cumulative
            
            # PG composition (typical)
            "caso4_fraction": 0.92,
            "p2o5_fraction": 0.01,
            "f_fraction": 0.006,
            "ra226_bq_kg": 500,
            
            # Leachate factors
            "leachate_m3_per_t": 0.1,
            "p_leach_fraction": 0.001,  # Fraction of P leaching
            "f_leach_fraction": 0.01,
        }
    
    def _build_inventory(self) -> LifeCycleInventory:
        """Build LCI for stack disposal."""
        p = self.parameters
        
        lci = LifeCycleInventory(
            process_name="Phosphogypsum Stack Disposal",
            functional_unit="1 kg PG",
            functional_unit_value=1.0
        )
        
        # Inputs
        lci.add_input("Phosphogypsum", 1.0, "kg")
        lci.add_input("Diesel", p["diesel_kg_per_t"] / 1000, "kg")
        lci.add_input("Electricity", p["electricity_kwh_per_t"] / 1000, "kWh")
        
        # Emissions to air
        diesel_co2 = (p["diesel_kg_per_t"] / 1000) * p["co2_factor"]
        lci.add_emission("CO2", diesel_co2, "kg", "air")
        
        # Radon emissions (Bq per kg PG over long term)
        annual_radon = p["radon_emission_rate"] * p["stack_area_m2_per_t"] * 365 * 24 * 3600
        lci.add_emission("Radon-222", annual_radon / 1000, "Bq", "air")  # Per kg input
        
        # Emissions to water (leachate)
        p_leach = p["p2o5_fraction"] * 0.436 * p["p_leach_fraction"]  # P2O5 to P
        lci.add_emission("Phosphorus", p_leach, "kg", "water")
        
        f_leach = p["f_fraction"] * p["f_leach_fraction"]
        lci.add_emission("Fluoride", f_leach, "kg", "water")
        
        return lci
    
    def get_capex_data(self) -> Dict:
        """CAPEX for stack construction."""
        # Based on EPA estimates and literature
        return {
            "equipment": [
                {"name": "Liner system", "cost": 500000, "installed": True},
                {"name": "Leachate collection", "cost": 200000, "installed": True},
                {"name": "Monitoring systems", "cost": 100000, "installed": True},
                {"name": "Transport equipment", "cost": 300000},
            ],
            "land": 1000000,  # Land cost for stack area
            "factors": {
                "engineering": 0.10,
                "contingency": 0.15,
            }
        }
    
    def get_opex_data(self) -> Dict:
        """OPEX for stack operation."""
        return {
            "materials": [],  # No chemical inputs
            "utilities": {
                "electricity_kwh": self.parameters["electricity_kwh_per_t"],
            },
            "labor": {
                "hours_per_tonne": 0.1,
            },
            "maintenance": 50000,  # Annual
            "overhead_rate": 0.10,
        }
    
    def get_products(self) -> List[Dict]:
        """No products from stack disposal."""
        return []


if __name__ == "__main__":
    pathway = StackDisposalPathway(country="USA")
    print(f"Pathway: {pathway.name} ({pathway.code})")
    print(f"TRL: {pathway.trl}")
    
    inv = pathway.get_scaled_inventory(1000)  # 1 tonne
    print(f"Inputs: {len(inv['inputs'])}")
    print(f"Air emissions: {len(inv['emissions_air'])}")
