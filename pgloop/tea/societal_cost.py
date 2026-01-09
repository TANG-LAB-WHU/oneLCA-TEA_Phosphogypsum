"""
Societal Cost Module

Calculates the total societal cost including internal and external costs.
"""

from typing import Dict


class SocietalCostCalculator:
    """Calculates SLCC (Societal Life Cycle Costing)."""
    
    def __init__(self, internal_cost: float, external_cost: float):
        self.internal_cost = internal_cost
        self.external_cost = external_cost
    
    def calculate_slcc(self) -> float:
        """SLCC = Internal Cost + External Cost."""
        return self.internal_cost + self.external_cost
    
    def get_breakdown(self) -> Dict[str, float]:
        """Return cost components."""
        total = self.calculate_slcc()
        if total == 0: return {}
        return {
            "internal_fraction": self.internal_cost / total,
            "external_fraction": self.external_cost / total,
            "total": total
        }
