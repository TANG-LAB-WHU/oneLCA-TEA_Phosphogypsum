"""
Impact Assessment Module

Calculates environmental impacts from LCI data.
"""

from typing import Any, Dict, List, Optional
from pgloop.lca.characterization import CharacterizationFactors


class ImpactAssessment:
    """
    Life Cycle Impact Assessment (LCIA) calculator.
    
    Implements ILCD recommended impact categories.
    """
    
    def __init__(self, characterization: CharacterizationFactors = None):
        """
        Initialize impact assessment.
        
        Args:
            characterization: CharacterizationFactors instance
        """
        self.cf = characterization or CharacterizationFactors()
    
    def calculate(self, inventory: Dict) -> Dict[str, float]:
        """
        Calculate impacts from inventory.
        
        Args:
            inventory: LCI as dictionary
            
        Returns:
            Dict of impact category -> value
        """
        impacts = {}
        
        # Process each impact category
        for category in self.cf.CATEGORIES:
            impacts[category] = self._calculate_category(inventory, category)
        
        return impacts
    
    def _calculate_category(
        self,
        inventory: Dict,
        category: str
    ) -> float:
        """Calculate a single impact category."""
        total = 0.0
        factors = self.cf.get_factors(category)
        
        # Emissions to air
        for emission in inventory.get("emissions_air", []):
            name = emission.get("name", "").lower()
            quantity = emission.get("quantity", 0)
            
            # Try exact match first, then partial match
            cf = factors.get(name, 0)
            if cf == 0:
                for factor_name, factor_value in factors.items():
                    if factor_name in name or name in factor_name:
                        cf = factor_value
                        break
            
            total += quantity * cf
        
        # Emissions to water
        for emission in inventory.get("emissions_water", []):
            name = emission.get("name", "").lower()
            quantity = emission.get("quantity", 0)
            
            cf = factors.get(f"water_{name}", factors.get(name, 0))
            total += quantity * cf
        
        # Emissions to soil
        for emission in inventory.get("emissions_soil", []):
            name = emission.get("name", "").lower()
            quantity = emission.get("quantity", 0)
            
            cf = factors.get(f"soil_{name}", factors.get(name, 0))
            total += quantity * cf
        
        # Avoided products (negative contribution)
        for avoided in inventory.get("avoided_products", []):
            name = avoided.get("name", "").lower()
            quantity = avoided.get("quantity", 0)
            
            cf = factors.get(f"avoided_{name}", 0)
            total -= quantity * cf  # Subtract avoided impacts
        
        return total
    
    def normalize(
        self,
        impacts: Dict[str, float],
        reference: str = "EU27"
    ) -> Dict[str, float]:
        """
        Normalize impacts to reference values.
        
        Args:
            impacts: Impact assessment results
            reference: Normalization reference (EU27, World, etc.)
            
        Returns:
            Normalized impacts (person-equivalents)
        """
        normalization_factors = self.cf.get_normalization_factors(reference)
        normalized = {}
        
        for category, value in impacts.items():
            nf = normalization_factors.get(category, 1)
            if nf != 0:
                normalized[category] = value / nf
            else:
                normalized[category] = 0
        
        return normalized
    
    def weight(
        self,
        normalized_impacts: Dict[str, float],
        weights: Dict[str, float] = None
    ) -> float:
        """
        Calculate single weighted score.
        
        Args:
            normalized_impacts: Normalized impact values
            weights: Weighting factors per category
            
        Returns:
            Single weighted score
        """
        if weights is None:
            # Equal weighting as default
            weights = {cat: 1.0 / len(normalized_impacts) for cat in normalized_impacts}
        
        total = sum(
            normalized_impacts.get(cat, 0) * weights.get(cat, 0)
            for cat in normalized_impacts
        )
        
        return total


if __name__ == "__main__":
    # Example usage
    from pgloop.lca.characterization import CharacterizationFactors
    
    cf = CharacterizationFactors()
    ia = ImpactAssessment(cf)
    
    inventory = {
        "emissions_air": [
            {"name": "CO2", "quantity": 100, "unit": "kg"},
            {"name": "CH4", "quantity": 2, "unit": "kg"},
        ],
        "emissions_water": [
            {"name": "P", "quantity": 0.1, "unit": "kg"},
        ],
        "emissions_soil": [],
        "avoided_products": []
    }
    
    impacts = ia.calculate(inventory)
    print("Impacts:", impacts)
