"""
Normalization Module

Normalizes LCA results against reference values.
"""

from typing import Dict, Optional


class LCANormalizer:
    """Normalizes impact categories against reference populations."""
    
    def __init__(self, reference: str = "EU27"):
        """
        Init with reference data.
        References: EU27 (2010), World (2010)
        Units: person-equivalents per year
        """
        self.reference = reference
        # Factors from ILCD/Product Environmental Footprint
        self.factors = {
            "EU27": {
                "climate_change": 1.31e-4,  # 1/7630 kg CO2-eq
                "acidification": 2.11e-2,   # 1/47.3 mol H+-eq
                "eutrophication_freshwater": 6.80e-1, # 1/1.47 kg P-eq
                "particulate_matter": 1.58e3,  # 1/0.000631 disease inc.
            },
            "World": {
                "climate_change": 1.13e-4,
                "acidification": 1.76e-2,
            }
        }
    
    def normalize(self, impacts: Dict[str, float]) -> Dict[str, float]:
        """Apply normalization factors to impacts."""
        normalized = {}
        ref_factors = self.factors.get(self.reference, self.factors["EU27"])
        
        for cat, value in impacts.items():
            factor = ref_factors.get(cat)
            if factor:
                normalized[cat] = value * factor
            else:
                normalized[cat] = 0.0
        
        return normalized
