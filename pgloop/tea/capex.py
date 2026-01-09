"""
CAPEX Calculator Module

Capital expenditure calculation and annualization.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class EquipmentCost:
    """Equipment cost entry."""
    
    name: str
    base_cost: float  # USD
    scaling_factor: float = 0.6  # For capacity scaling
    installation_factor: float = 1.3  # Installed cost = base * factor
    base_capacity: float = 1.0
    capacity_unit: str = "t/h"


class CAPEXCalculator:
    """
    Capital Expenditure Calculator.
    
    Handles:
    - Equipment costs with scaling
    - Installation factors
    - Indirect costs (engineering, contingency)
    - Annualization
    """
    
    # Default factors
    DEFAULT_FACTORS = {
        "installation": 1.3,
        "instrumentation": 0.1,
        "piping": 0.15,
        "electrical": 0.1,
        "buildings": 0.15,
        "site_prep": 0.05,
        "engineering": 0.15,
        "contingency": 0.15,
    }
    
    def __init__(self, params: Dict = None):
        """
        Initialize CAPEX calculator.
        
        Args:
            params: Economic parameters (discount_rate, lifetime)
        """
        self.params = params or {}
        self.factors = self.DEFAULT_FACTORS.copy()
    
    def calculate_equipment_cost(
        self,
        base_cost: float,
        base_capacity: float,
        target_capacity: float,
        scaling_exponent: float = 0.6
    ) -> float:
        """
        Scale equipment cost using power law.
        
        Cost_target = Cost_base * (Capacity_target / Capacity_base)^n
        
        Args:
            base_cost: Reference equipment cost
            base_capacity: Reference capacity
            target_capacity: Target capacity
            scaling_exponent: Usually 0.6 for chemical equipment
            
        Returns:
            Scaled equipment cost
        """
        if base_capacity <= 0:
            return base_cost
        
        return base_cost * (target_capacity / base_capacity) ** scaling_exponent
    
    def calculate_installed_cost(
        self,
        equipment_cost: float,
        installation_factor: float = None
    ) -> float:
        """
        Calculate installed equipment cost.
        
        Args:
            equipment_cost: Bare equipment cost
            installation_factor: Installation multiplier
            
        Returns:
            Installed cost
        """
        factor = installation_factor or self.factors["installation"]
        return equipment_cost * factor
    
    def calculate_total(self, capex_data: Dict) -> float:
        """
        Calculate total CAPEX from component data.
        
        Args:
            capex_data: Dict with equipment costs and factors
            
        Returns:
            Total capital expenditure
        """
        equipment_list = capex_data.get("equipment", [])
        
        # Sum equipment costs
        total_equipment = 0.0
        for eq in equipment_list:
            if isinstance(eq, dict):
                cost = eq.get("cost", 0)
                installed = eq.get("installed", False)
                if not installed:
                    cost = self.calculate_installed_cost(cost)
                total_equipment += cost
            else:
                total_equipment += eq
        
        # Apply indirect cost factors
        factors = capex_data.get("factors", self.factors)
        
        indirect_costs = 0.0
        for factor_name in ["instrumentation", "piping", "electrical", 
                           "buildings", "site_prep"]:
            indirect_costs += total_equipment * factors.get(factor_name, 0)
        
        direct_capital = total_equipment + indirect_costs
        
        # Engineering and contingency
        engineering = direct_capital * factors.get("engineering", 0.15)
        contingency = direct_capital * factors.get("contingency", 0.15)
        
        total_capex = direct_capital + engineering + contingency
        
        # Add land if specified
        land = capex_data.get("land", 0)
        total_capex += land
        
        return total_capex
    
    def annualize(
        self,
        total_capex: float,
        discount_rate: float = None,
        lifetime: int = None
    ) -> float:
        """
        Convert CAPEX to annualized cost.
        
        A = P * [r(1+r)^n] / [(1+r)^n - 1]
        
        Args:
            total_capex: Total capital cost
            discount_rate: Annual discount rate
            lifetime: Project lifetime in years
            
        Returns:
            Annualized capital cost
        """
        r = discount_rate or self.params.get("discount_rate", 0.05)
        n = lifetime or self.params.get("lifetime_years", 20)
        
        if r == 0:
            return total_capex / n
        
        factor = (r * (1 + r) ** n) / ((1 + r) ** n - 1)
        return total_capex * factor
    
    def calculate_breakdown(self, capex_data: Dict) -> Dict[str, float]:
        """Get detailed CAPEX breakdown."""
        equipment_list = capex_data.get("equipment", [])
        factors = capex_data.get("factors", self.factors)
        
        breakdown = {"equipment": {}, "indirect": {}}
        
        total_equipment = 0.0
        for eq in equipment_list:
            if isinstance(eq, dict):
                name = eq.get("name", "unknown")
                cost = eq.get("cost", 0)
                if not eq.get("installed", False):
                    cost = self.calculate_installed_cost(cost)
                breakdown["equipment"][name] = cost
                total_equipment += cost
        
        for factor_name in ["instrumentation", "piping", "electrical",
                           "buildings", "site_prep", "engineering", "contingency"]:
            if factor_name in factors:
                breakdown["indirect"][factor_name] = total_equipment * factors[factor_name]
        
        breakdown["total_equipment"] = total_equipment
        breakdown["total_indirect"] = sum(breakdown["indirect"].values())
        breakdown["land"] = capex_data.get("land", 0)
        breakdown["total"] = (breakdown["total_equipment"] + 
                             breakdown["total_indirect"] + 
                             breakdown["land"])
        
        return breakdown


if __name__ == "__main__":
    calc = CAPEXCalculator({"discount_rate": 0.05, "lifetime_years": 20})
    
    capex_data = {
        "equipment": [
            {"name": "Reactor", "cost": 500000},
            {"name": "Conveyor", "cost": 100000},
        ],
        "land": 50000
    }
    
    total = calc.calculate_total(capex_data)
    annual = calc.annualize(total)
    
    print(f"Total CAPEX: ${total:,.0f}")
    print(f"Annualized: ${annual:,.0f}/year")
