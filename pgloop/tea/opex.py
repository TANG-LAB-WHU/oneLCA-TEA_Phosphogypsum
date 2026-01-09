"""
OPEX Calculator Module

Operational expenditure calculation.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


# Default unit prices by country (USD)
UNIT_PRICES = {
    "global": {
        "electricity": 0.10,  # USD/kWh
        "natural_gas": 0.008,  # USD/MJ
        "water": 0.002,  # USD/kg
        "h2so4": 0.07,  # USD/kg
        "hcl": 0.15,  # USD/kg
        "naoh": 0.35,  # USD/kg
        "lime": 0.10,  # USD/kg
        "labor": 25.0,  # USD/hour
    },
    "China": {
        "electricity": 0.08,
        "natural_gas": 0.015,
        "water": 0.0005,
        "h2so4": 0.05,
        "hcl": 0.10,
        "naoh": 0.25,
        "lime": 0.04,
        "labor": 8.0,
    },
    "USA": {
        "electricity": 0.12,
        "natural_gas": 0.005,
        "water": 0.003,
        "h2so4": 0.08,
        "hcl": 0.18,
        "naoh": 0.40,
        "lime": 0.08,
        "labor": 35.0,
    },
    "Morocco": {
        "electricity": 0.11,
        "natural_gas": 0.010,
        "water": 0.001,
        "h2so4": 0.06,
        "hcl": 0.12,
        "naoh": 0.30,
        "lime": 0.05,
        "labor": 6.0,
    },
    "EU": {
        "electricity": 0.15,
        "natural_gas": 0.012,
        "water": 0.004,
        "h2so4": 0.10,
        "hcl": 0.20,
        "naoh": 0.45,
        "lime": 0.10,
        "labor": 40.0,
    }
}


class OPEXCalculator:
    """
    Operational Expenditure Calculator.
    
    Categories:
    - Raw materials
    - Utilities (electricity, heat, water)
    - Labor
    - Maintenance
    - Overhead
    """
    
    def __init__(self, params: Dict = None, country: str = "global"):
        """
        Initialize OPEX calculator.
        
        Args:
            params: Economic parameters
            country: Country for unit prices
        """
        self.params = params or {}
        self.country = country
        self.unit_prices = UNIT_PRICES.get(country, UNIT_PRICES["global"]).copy()
    
    def set_unit_price(self, item: str, price: float) -> None:
        """Set custom unit price."""
        self.unit_prices[item.lower()] = price
    
    def calculate(
        self,
        opex_data: Dict,
        functional_unit_kg: float = 1000
    ) -> Dict:
        """
        Calculate operational costs.
        
        Args:
            opex_data: Dict with consumption data
            functional_unit_kg: Functional unit in kg
            
        Returns:
            Dict with total and breakdown
        """
        breakdown = {}
        total = 0.0
        
        # Raw materials
        materials = opex_data.get("materials", [])
        materials_cost = 0.0
        
        for mat in materials:
            name = mat.get("name", "").lower()
            quantity = mat.get("quantity", 0)
            unit = mat.get("unit", "kg")
            custom_price = mat.get("price")
            
            # Get unit price
            price = custom_price if custom_price is not None else self.unit_prices.get(name, 0)
            
            # Scale quantity to functional unit
            scaled_qty = quantity * (functional_unit_kg / mat.get("per_kg_input", 1000))
            
            cost = scaled_qty * price
            materials_cost += cost
            breakdown[f"material_{name}"] = cost
        
        total += materials_cost
        breakdown["materials_total"] = materials_cost
        
        # Utilities
        utilities = opex_data.get("utilities", {})
        utilities_cost = 0.0
        
        # Electricity
        electricity_kwh = utilities.get("electricity_kwh", 0)
        scaled_elec = electricity_kwh * (functional_unit_kg / 1000)
        elec_cost = scaled_elec * self.unit_prices.get("electricity", 0.10)
        utilities_cost += elec_cost
        breakdown["electricity"] = elec_cost
        
        # Heat/Gas
        heat_mj = utilities.get("heat_mj", 0)
        scaled_heat = heat_mj * (functional_unit_kg / 1000)
        heat_cost = scaled_heat * self.unit_prices.get("natural_gas", 0.008)
        utilities_cost += heat_cost
        breakdown["heat"] = heat_cost
        
        # Water
        water_kg = utilities.get("water_kg", 0)
        scaled_water = water_kg * (functional_unit_kg / 1000)
        water_cost = scaled_water * self.unit_prices.get("water", 0.002)
        utilities_cost += water_cost
        breakdown["water"] = water_cost
        
        total += utilities_cost
        breakdown["utilities_total"] = utilities_cost
        
        # Labor
        labor = opex_data.get("labor", {})
        labor_hours = labor.get("hours_per_tonne", 0) * (functional_unit_kg / 1000)
        labor_rate = labor.get("rate", self.unit_prices.get("labor", 25))
        labor_cost = labor_hours * labor_rate
        total += labor_cost
        breakdown["labor"] = labor_cost
        
        # Maintenance (typically 2-4% of CAPEX annually)
        maintenance = opex_data.get("maintenance", 0)
        total += maintenance * (functional_unit_kg / 1000)
        breakdown["maintenance"] = maintenance * (functional_unit_kg / 1000)
        
        # Overhead (typically 15-25% of operating costs)
        overhead_rate = opex_data.get("overhead_rate", 0.15)
        overhead = total * overhead_rate
        total += overhead
        breakdown["overhead"] = overhead
        
        return {
            "total": total,
            "breakdown": breakdown,
            "per_tonne": total / (functional_unit_kg / 1000)
        }
    
    def estimate_from_capex(
        self,
        capex_total: float,
        annual_throughput: float
    ) -> Dict:
        """
        Estimate OPEX from CAPEX using typical ratios.
        
        Args:
            capex_total: Total capital cost
            annual_throughput: Tonnes per year
            
        Returns:
            Estimated OPEX breakdown
        """
        # Typical annual cost percentages of CAPEX
        estimates = {
            "maintenance": capex_total * 0.03,  # 3%
            "insurance": capex_total * 0.01,  # 1%
            "taxes": capex_total * 0.02,  # 2%
        }
        
        fixed_annual = sum(estimates.values())
        per_tonne = fixed_annual / annual_throughput if annual_throughput > 0 else 0
        
        return {
            "fixed_annual": fixed_annual,
            "per_tonne_fixed": per_tonne,
            "breakdown": estimates
        }


if __name__ == "__main__":
    calc = OPEXCalculator(country="China")
    
    opex_data = {
        "materials": [
            {"name": "H2SO4", "quantity": 50, "unit": "kg", "per_kg_input": 1000},
            {"name": "Lime", "quantity": 30, "unit": "kg", "per_kg_input": 1000},
        ],
        "utilities": {
            "electricity_kwh": 100,
            "water_kg": 500,
        },
        "labor": {
            "hours_per_tonne": 0.5,
        }
    }
    
    result = calc.calculate(opex_data, 1000)
    print(f"Total OPEX: ${result['total']:.2f}/tonne")
    for key, value in result['breakdown'].items():
        print(f"  {key}: ${value:.2f}")
