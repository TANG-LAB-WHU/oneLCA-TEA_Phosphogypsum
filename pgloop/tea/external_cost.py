"""
External Cost Calculator Module

Calculates external costs from emissions using shadow prices.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import json


# Default shadow prices (EUR, converted to USD at 1.1)
# Based on CE Delft Environmental Prices Handbook
DEFAULT_SHADOW_PRICES = {
    # Air emissions (USD/kg)
    "co2": 0.055,  # ~50 EUR/tonne CO2
    "carbon_dioxide": 0.055,
    "ch4": 1.375,  # 25x CO2
    "methane": 1.375,
    "n2o": 16.4,  # 298x CO2
    "nitrous_oxide": 16.4,
    "so2": 12.1,
    "sulfur_dioxide": 12.1,
    "nox": 8.8,
    "nitrogen_oxides": 8.8,
    "nh3": 11.0,
    "ammonia": 11.0,
    "pm2.5": 40.7,
    "pm10": 16.5,
    "particulate_matter": 40.7,
    "voc": 2.2,
    "nmvoc": 2.2,
    
    # Water emissions (USD/kg)
    "water_p": 66.0,  # Phosphorus to freshwater
    "water_phosphorus": 66.0,
    "water_n": 3.3,  # Nitrogen to marine
    "water_nitrogen": 3.3,
    "water_cod": 0.55,
    "water_bod": 0.55,
    
    # Soil emissions - Heavy metals (USD/kg)
    "soil_cd": 27500,  # Cadmium
    "soil_cadmium": 27500,
    "soil_pb": 550,  # Lead
    "soil_lead": 550,
    "soil_hg": 55000,  # Mercury
    "soil_mercury": 55000,
    "soil_as": 16500,  # Arsenic
    "soil_arsenic": 16500,
    "soil_cr": 1100,  # Chromium
    "soil_chromium": 1100,
    "soil_ni": 550,  # Nickel
    "soil_nickel": 550,
    "soil_zn": 55,  # Zinc
    "soil_zinc": 55,
    "soil_cu": 110,  # Copper
    "soil_copper": 110,
    
    # Radioactive emissions (USD/Bq)
    "ra226": 0.001,  # Radium-226
    "radium_226": 0.001,
    "rn222": 0.0001,  # Radon-222
    "radon_222": 0.0001,
    
    # Avoided production (negative = benefit)
    # These represent avoided external costs
    "avoided_electricity_kwh": -0.05,  # Coal mix
    "avoided_natural_gas_mj": -0.003,
    "avoided_fertilizer_n_kg": -0.5,  # Avoided N fertilizer production
    "avoided_fertilizer_p_kg": -0.3,
    "avoided_gypsum_kg": -0.02,
}


class ExternalCostCalculator:
    """
    External Cost Calculator for Societal LCC.
    
    Calculates environmental externalities using shadow prices.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize external cost calculator.
        
        Args:
            config_path: Path to custom shadow prices
        """
        self.config_path = config_path
        self.prices = DEFAULT_SHADOW_PRICES.copy()
        
        if config_path:
            self._load_custom_prices()
    
    def _load_custom_prices(self) -> None:
        """Load custom shadow prices from config."""
        prices_file = self.config_path / "shadow_prices.yaml"
        
        if prices_file.exists():
            try:
                import yaml
                with open(prices_file, "r") as f:
                    custom = yaml.safe_load(f)
                    if custom:
                        self.prices.update(custom)
            except ImportError:
                # Try JSON
                json_file = self.config_path / "shadow_prices.json"
                if json_file.exists():
                    with open(json_file, "r") as f:
                        custom = json.load(f)
                        self.prices.update(custom)
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update shadow prices."""
        self.prices.update(prices)
    
    def calculate(self, emissions: Dict) -> Dict:
        """
        Calculate external costs from emissions.
        
        Args:
            emissions: Dict with emissions_air, emissions_water, emissions_soil
            
        Returns:
            Dict with total and breakdown
        """
        breakdown = {}
        total = 0.0
        
        # Air emissions
        for emission in emissions.get("emissions_air", []):
            name = emission.get("name", "").lower().replace(" ", "_")
            quantity = emission.get("quantity", 0)
            unit = emission.get("unit", "kg")
            
            # Handle Bq for radioactive
            if unit.lower() == "bq":
                price = self.prices.get(name, self.prices.get(f"air_{name}", 0))
            else:
                price = self.prices.get(name, self.prices.get(f"air_{name}", 0))
            
            cost = quantity * price
            if cost != 0:
                breakdown[f"air_{name}"] = cost
                total += cost
        
        # Water emissions
        for emission in emissions.get("emissions_water", []):
            name = emission.get("name", "").lower().replace(" ", "_")
            quantity = emission.get("quantity", 0)
            
            price = self.prices.get(f"water_{name}", self.prices.get(name, 0))
            cost = quantity * price
            if cost != 0:
                breakdown[f"water_{name}"] = cost
                total += cost
        
        # Soil emissions
        for emission in emissions.get("emissions_soil", []):
            name = emission.get("name", "").lower().replace(" ", "_")
            quantity = emission.get("quantity", 0)
            
            price = self.prices.get(f"soil_{name}", self.prices.get(name, 0))
            cost = quantity * price
            if cost != 0:
                breakdown[f"soil_{name}"] = cost
                total += cost
        
        # Avoided products (benefits = negative costs)
        for avoided in emissions.get("avoided_products", []):
            name = avoided.get("name", "").lower().replace(" ", "_")
            quantity = avoided.get("quantity", 0)
            
            price = self.prices.get(f"avoided_{name}", 0)
            benefit = quantity * price  # Will be negative
            if benefit != 0:
                breakdown[f"avoided_{name}"] = benefit
                total += benefit
        
        return {
            "total": total,
            "breakdown": breakdown,
            "costs": {k: v for k, v in breakdown.items() if v > 0},
            "benefits": {k: v for k, v in breakdown.items() if v < 0}
        }
    
    def calculate_from_lca(
        self,
        lca_result,  # LCAResult
        functional_unit_kg: float = 1000
    ) -> Dict:
        """
        Calculate external costs from LCA inventory.
        
        Args:
            lca_result: LCAResult object
            functional_unit_kg: Functional unit in kg
            
        Returns:
            External cost calculation
        """
        emissions = {
            "emissions_air": lca_result.inventory.get("emissions_air", []),
            "emissions_water": lca_result.inventory.get("emissions_water", []),
            "emissions_soil": lca_result.inventory.get("emissions_soil", []),
            "avoided_products": lca_result.inventory.get("avoided_products", [])
        }
        
        return self.calculate(emissions)
    
    def get_price(self, emission_name: str, compartment: str = "") -> float:
        """Get shadow price for a specific emission."""
        name = emission_name.lower().replace(" ", "_")
        
        if compartment:
            key = f"{compartment}_{name}"
            if key in self.prices:
                return self.prices[key]
        
        return self.prices.get(name, 0)
    
    def save_prices(self, filepath: Path) -> None:
        """Save current prices to file."""
        with open(filepath, "w") as f:
            json.dump(self.prices, f, indent=2)


if __name__ == "__main__":
    calc = ExternalCostCalculator()
    
    emissions = {
        "emissions_air": [
            {"name": "CO2", "quantity": 100, "unit": "kg"},
            {"name": "SO2", "quantity": 0.5, "unit": "kg"},
            {"name": "Ra226", "quantity": 1000, "unit": "Bq"},
        ],
        "emissions_water": [
            {"name": "P", "quantity": 0.1, "unit": "kg"},
        ],
        "emissions_soil": [
            {"name": "Cd", "quantity": 0.001, "unit": "kg"},
        ],
        "avoided_products": []
    }
    
    result = calc.calculate(emissions)
    print(f"Total External Cost: ${result['total']:.2f}")
    for key, value in result['breakdown'].items():
        print(f"  {key}: ${value:.4f}")
