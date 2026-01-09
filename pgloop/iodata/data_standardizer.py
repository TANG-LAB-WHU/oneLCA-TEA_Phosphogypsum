"""
Data Standardizer Module

Converts heterogeneous data from various sources into a standardized format
for use in LCA and TEA calculations.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import re


# Standard units for LCA/TEA
STANDARD_UNITS = {
    # Mass
    "mass": "kg",
    "t": 1000,  # kg
    "tonne": 1000,
    "ton": 1000,
    "mt": 1e6,  # kg (million tonnes)
    "g": 0.001,
    "mg": 1e-6,
    
    # Energy
    "energy": "MJ",
    "kwh": 3.6,  # MJ
    "mwh": 3600,
    "gj": 1000,
    "tj": 1e6,
    "btu": 0.001055,
    
    # Volume
    "volume": "m3",
    "l": 0.001,
    "ml": 1e-6,
    "gal": 0.003785,
    
    # Radioactivity
    "radioactivity": "Bq/kg",
    "pci/g": 37,  # Bq/kg (1 pCi/g = 37 Bq/kg)
    
    # Currency
    "currency": "USD",
    "eur": 1.1,  # approximate USD
    "cny": 0.14,  # approximate USD
    "mad": 0.1,  # approximate USD
}


@dataclass
class StandardizedData:
    """Standardized data record for LCI/TEA."""
    
    parameter: str
    value: float
    unit: str
    uncertainty_min: Optional[float] = None
    uncertainty_max: Optional[float] = None
    distribution: str = "triangular"
    source: str = ""
    quality_score: float = 1.0  # 0-1, higher is better
    country: str = "global"
    year: int = 2024
    notes: str = ""


class DataStandardizer:
    """
    Standardizes data from various sources into consistent formats.
    
    Features:
    - Unit conversion
    - Uncertainty handling
    - Quality scoring
    - Source tracking
    """
    
    def __init__(self, base_year: int = 2024, base_currency: str = "USD"):
        """
        Initialize the data standardizer.
        
        Args:
            base_year: Base year for temporal adjustments
            base_currency: Base currency for cost data
        """
        self.base_year = base_year
        self.base_currency = base_currency
        self.conversion_factors = STANDARD_UNITS.copy()
    
    def convert_unit(
        self, 
        value: float, 
        from_unit: str, 
        to_unit: str
    ) -> float:
        """
        Convert a value from one unit to another.
        
        Args:
            value: The value to convert
            from_unit: Source unit
            to_unit: Target unit
            
        Returns:
            Converted value
        """
        from_unit = from_unit.lower().strip()
        to_unit = to_unit.lower().strip()
        
        if from_unit == to_unit:
            return value
        
        # Get conversion factors
        from_factor = self.conversion_factors.get(from_unit, 1.0)
        to_factor = self.conversion_factors.get(to_unit, 1.0)
        
        if isinstance(from_factor, str) or isinstance(to_factor, str):
            raise ValueError(f"Cannot convert between {from_unit} and {to_unit}")
        
        return value * from_factor / to_factor
    
    def standardize_composition(
        self, 
        data: Dict[str, Any],
        source: str = "",
        country: str = "global"
    ) -> List[StandardizedData]:
        """
        Standardize phosphogypsum composition data.
        
        Args:
            data: Raw composition data dict
            source: Data source reference
            country: Country of origin
            
        Returns:
            List of standardized data records
        """
        standardized = []
        
        # Common composition parameters
        composition_params = {
            "CaSO4": "mass_fraction",
            "P2O5": "mass_fraction",
            "F": "mass_fraction",
            "SiO2": "mass_fraction",
            "Fe2O3": "mass_fraction",
            "Al2O3": "mass_fraction",
            "moisture": "mass_fraction",
            "Ra226": "Bq/kg",
            "Cd": "mg/kg",
            "Pb": "mg/kg",
            "As": "mg/kg",
            "Hg": "mg/kg",
        }
        
        for param, unit in composition_params.items():
            if param in data:
                value = data[param]
                
                # Handle range values
                if isinstance(value, (list, tuple)):
                    val = sum(value) / len(value)
                    uncertainty_min = min(value)
                    uncertainty_max = max(value)
                else:
                    val = float(value)
                    uncertainty_min = None
                    uncertainty_max = None
                
                standardized.append(StandardizedData(
                    parameter=param,
                    value=val,
                    unit=unit,
                    uncertainty_min=uncertainty_min,
                    uncertainty_max=uncertainty_max,
                    source=source,
                    country=country
                ))
        
        return standardized
    
    def standardize_lci(
        self,
        inputs: Dict[str, Dict],
        outputs: Dict[str, Dict],
        emissions: Dict[str, Dict],
        source: str = ""
    ) -> Dict[str, List[StandardizedData]]:
        """
        Standardize Life Cycle Inventory data.
        
        Args:
            inputs: Input materials/energy
            outputs: Output products
            emissions: Emissions to air/water/soil
            source: Data source reference
            
        Returns:
            Dict with standardized inputs, outputs, emissions
        """
        result = {
            "inputs": [],
            "outputs": [],
            "emissions": []
        }
        
        for name, data in inputs.items():
            result["inputs"].append(self._standardize_flow(name, data, source))
        
        for name, data in outputs.items():
            result["outputs"].append(self._standardize_flow(name, data, source))
        
        for name, data in emissions.items():
            result["emissions"].append(self._standardize_flow(name, data, source))
        
        return result
    
    def _standardize_flow(
        self, 
        name: str, 
        data: Dict, 
        source: str
    ) -> StandardizedData:
        """Standardize a single flow entry."""
        value = data.get("value", 0)
        unit = data.get("unit", "kg")
        
        # Convert to standard units if possible
        standard_unit = self._get_standard_unit(unit)
        if standard_unit != unit:
            try:
                value = self.convert_unit(value, unit, standard_unit)
                unit = standard_unit
            except ValueError:
                pass  # Keep original unit if conversion fails
        
        return StandardizedData(
            parameter=name,
            value=value,
            unit=unit,
            uncertainty_min=data.get("min"),
            uncertainty_max=data.get("max"),
            distribution=data.get("distribution", "triangular"),
            source=source,
            quality_score=data.get("quality", 1.0)
        )
    
    def _get_standard_unit(self, unit: str) -> str:
        """Get the standard unit for a given unit type."""
        unit_lower = unit.lower()
        
        # Mass units
        if unit_lower in ["kg", "t", "tonne", "ton", "g", "mg"]:
            return "kg"
        
        # Energy units
        if unit_lower in ["mj", "kwh", "gj", "tj"]:
            return "MJ"
        
        # Volume units
        if unit_lower in ["m3", "l", "ml"]:
            return "m3"
        
        return unit  # Return original if no mapping
    
    def calculate_quality_score(
        self,
        source_type: str,
        year: int,
        geographic_match: bool,
        technology_match: bool
    ) -> float:
        """
        Calculate a data quality score based on pedigree criteria.
        
        Args:
            source_type: Type of source (primary, secondary, estimated)
            year: Year of the data
            geographic_match: Whether geography matches
            technology_match: Whether technology matches
            
        Returns:
            Quality score between 0 and 1
        """
        score = 1.0
        
        # Source type factor
        source_factors = {
            "primary": 1.0,
            "secondary": 0.8,
            "literature": 0.6,
            "estimated": 0.4,
            "default": 0.2
        }
        score *= source_factors.get(source_type, 0.5)
        
        # Temporal factor
        age = self.base_year - year
        if age <= 3:
            score *= 1.0
        elif age <= 6:
            score *= 0.9
        elif age <= 10:
            score *= 0.7
        else:
            score *= 0.5
        
        # Geographic factor
        if geographic_match:
            score *= 1.0
        else:
            score *= 0.7
        
        # Technology factor
        if technology_match:
            score *= 1.0
        else:
            score *= 0.6
        
        return round(score, 3)


if __name__ == "__main__":
    # Example usage
    standardizer = DataStandardizer()
    
    # Example composition data
    composition = {
        "CaSO4": 0.92,
        "P2O5": [0.005, 0.015],  # range
        "Ra226": 500,
        "Cd": 0.5
    }
    
    result = standardizer.standardize_composition(
        composition,
        source="Literature Review 2024",
        country="China"
    )
    
    for item in result:
        print(f"{item.parameter}: {item.value} {item.unit}")
