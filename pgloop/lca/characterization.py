"""
Characterization Factors Module

Contains characterization factors for LCIA calculations.
Based on ILCD recommendations and open-source data.
"""

from typing import Any, Dict, Optional
from pathlib import Path
import json


class CharacterizationFactors:
    """
    Characterization factors for Life Cycle Impact Assessment.
    
    Categories (ILCD recommended):
    - climate_change: kg CO2-eq (IPCC GWP100)
    - acidification: mol H+-eq
    - eutrophication_fresh: kg P-eq
    - eutrophication_marine: kg N-eq
    - human_toxicity_cancer: CTUh
    - human_toxicity_noncancer: CTUh
    - ecotoxicity_freshwater: CTUe
    - ionizing_radiation: kBq U-235 eq
    - particulate_matter: disease incidence
    - resource_depletion: kg Sb-eq
    """
    
    CATEGORIES = [
        "climate_change",
        "acidification",
        "eutrophication_fresh",
        "eutrophication_marine",
        "human_toxicity_cancer",
        "human_toxicity_noncancer",
        "ecotoxicity_freshwater",
        "ionizing_radiation",
        "particulate_matter",
        "resource_depletion",
    ]
    
    # Default characterization factors (subset of common substances)
    # Units are per kg emission unless otherwise noted
    DEFAULT_FACTORS = {
        "climate_change": {
            # GWP100 (kg CO2-eq / kg)
            "co2": 1.0,
            "carbon dioxide": 1.0,
            "ch4": 28.0,
            "methane": 28.0,
            "n2o": 265.0,
            "nitrous oxide": 265.0,
            "sf6": 23500.0,
            "hfc-134a": 1300.0,
            # Biogenic CO2 (often counted as neutral)
            "co2_biogenic": 0.0,
        },
        "acidification": {
            # mol H+-eq / kg
            "so2": 31.2,
            "sulfur dioxide": 31.2,
            "nox": 21.7,
            "nitrogen oxides": 21.7,
            "no2": 21.7,
            "nh3": 58.8,
            "ammonia": 58.8,
            "hcl": 27.4,
            "hf": 50.0,
            "hydrogen fluoride": 50.0,
        },
        "eutrophication_fresh": {
            # kg P-eq / kg
            "p": 1.0,
            "phosphorus": 1.0,
            "phosphate": 0.326,
            "po4": 0.326,
            "p2o5": 0.436,
        },
        "eutrophication_marine": {
            # kg N-eq / kg
            "n": 1.0,
            "nitrogen": 1.0,
            "no3": 0.226,
            "nitrate": 0.226,
            "nh4": 0.778,
            "ammonium": 0.778,
            "nh3": 0.823,
        },
        "human_toxicity_cancer": {
            # CTUh / kg (comparative toxic units)
            "arsenic": 1.4e-4,
            "as": 1.4e-4,
            "cadmium": 1.1e-4,
            "cd": 1.1e-4,
            "chromium vi": 1.3e-3,
            "lead": 5.4e-7,
            "pb": 5.4e-7,
            "mercury": 3.4e-5,
            "hg": 3.4e-5,
            "nickel": 1.7e-5,
            "ni": 1.7e-5,
            "benzene": 2.8e-6,
            "formaldehyde": 1.3e-5,
            "dioxins": 5.3e-2,
        },
        "human_toxicity_noncancer": {
            # CTUh / kg
            "arsenic": 3.9e-4,
            "as": 3.9e-4,
            "cadmium": 1.8e-5,
            "cd": 1.8e-5,
            "lead": 8.6e-6,
            "pb": 8.6e-6,
            "mercury": 4.2e-4,
            "hg": 4.2e-4,
            "zinc": 1.3e-7,
            "zn": 1.3e-7,
            "fluoride": 2.5e-6,
            "f": 2.5e-6,
            "pm2.5": 6.3e-4,
            "particulate matter": 6.3e-4,
        },
        "ecotoxicity_freshwater": {
            # CTUe / kg
            "copper": 4.8e3,
            "cu": 4.8e3,
            "zinc": 9.2e2,
            "zn": 9.2e2,
            "nickel": 3.3e3,
            "ni": 3.3e3,
            "cadmium": 3.1e4,
            "cd": 3.1e4,
            "lead": 1.3e3,
            "pb": 1.3e3,
            "mercury": 8.1e4,
            "hg": 8.1e4,
            "arsenic": 1.2e4,
            "as": 1.2e4,
        },
        "ionizing_radiation": {
            # kBq U-235 eq / Bq
            "ra-226": 5.0e-3,
            "ra226": 5.0e-3,
            "radium-226": 5.0e-3,
            "rn-222": 5.9e-6,
            "rn222": 5.9e-6,
            "radon-222": 5.9e-6,
            "u-238": 1.2e-3,
            "uranium-238": 1.2e-3,
            "th-232": 1.5e-3,
            "thorium-232": 1.5e-3,
            "po-210": 4.3e-3,
            "polonium-210": 4.3e-3,
        },
        "particulate_matter": {
            # disease incidence / kg
            "pm2.5": 6.3e-4,
            "pm10": 2.5e-4,
            "dust": 2.5e-4,
            "so2": 5.4e-5,
            "nox": 2.3e-5,
            "nh3": 3.5e-5,
        },
        "resource_depletion": {
            # kg Sb-eq / kg
            "phosphate rock": 8.4e-5,
            "gypsum": 1.0e-6,
            "limestone": 1.0e-7,
            "iron ore": 5.0e-8,
            "copper ore": 1.4e-3,
            "zinc ore": 9.0e-4,
        },
    }
    
    # Normalization factors (per person per year)
    NORMALIZATION_FACTORS = {
        "EU27": {
            "climate_change": 8100,  # kg CO2-eq
            "acidification": 47.3,   # mol H+-eq
            "eutrophication_fresh": 1.61,  # kg P-eq
            "eutrophication_marine": 21.1,  # kg N-eq
            "human_toxicity_cancer": 3.4e-5,  # CTUh
            "human_toxicity_noncancer": 5.4e-4,  # CTUh
            "ecotoxicity_freshwater": 9260,  # CTUe
            "ionizing_radiation": 1030,  # kBq U-235 eq
            "particulate_matter": 5.8e-4,  # disease incidence
            "resource_depletion": 0.037,  # kg Sb-eq
        },
        "World": {
            "climate_change": 6500,
            "acidification": 38.0,
            "eutrophication_fresh": 1.3,
            "eutrophication_marine": 17.0,
            "human_toxicity_cancer": 2.8e-5,
            "human_toxicity_noncancer": 4.4e-4,
            "ecotoxicity_freshwater": 7500,
            "ionizing_radiation": 850,
            "particulate_matter": 4.7e-4,
            "resource_depletion": 0.030,
        }
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize characterization factors.
        
        Args:
            config_path: Path to custom factors configuration
        """
        self.factors = self.DEFAULT_FACTORS.copy()
        self.normalization = self.NORMALIZATION_FACTORS.copy()
        
        # Load custom factors if provided
        if config_path:
            self._load_custom_factors(config_path)
    
    def _load_custom_factors(self, config_path: Path) -> None:
        """Load custom characterization factors from config."""
        factors_file = config_path / "impact_factors.yaml"
        
        if factors_file.exists():
            try:
                import yaml
                with open(factors_file, "r") as f:
                    custom = yaml.safe_load(f)
                    if custom:
                        for category, factors in custom.items():
                            if category in self.factors:
                                self.factors[category].update(factors)
            except ImportError:
                # Try JSON fallback
                factors_file = config_path / "impact_factors.json"
                if factors_file.exists():
                    with open(factors_file, "r") as f:
                        custom = json.load(f)
                        for category, factors in custom.items():
                            if category in self.factors:
                                self.factors[category].update(factors)
    
    def get_factors(self, category: str) -> Dict[str, float]:
        """
        Get characterization factors for a category.
        
        Args:
            category: Impact category name
            
        Returns:
            Dict of substance -> characterization factor
        """
        return self.factors.get(category, {})
    
    def get_factor(
        self,
        category: str,
        substance: str
    ) -> float:
        """
        Get specific characterization factor.
        
        Args:
            category: Impact category
            substance: Substance name
            
        Returns:
            Characterization factor value
        """
        return self.factors.get(category, {}).get(substance.lower(), 0)
    
    def get_normalization_factors(
        self,
        reference: str = "EU27"
    ) -> Dict[str, float]:
        """
        Get normalization factors.
        
        Args:
            reference: Reference region (EU27, World)
            
        Returns:
            Dict of category -> normalization factor
        """
        return self.normalization.get(reference, {})
    
    def add_factor(
        self,
        category: str,
        substance: str,
        value: float
    ) -> None:
        """Add or update a characterization factor."""
        if category not in self.factors:
            self.factors[category] = {}
        self.factors[category][substance.lower()] = value
    
    def save(self, filepath: Path) -> None:
        """Save factors to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.factors, f, indent=2)


if __name__ == "__main__":
    cf = CharacterizationFactors()
    
    print("Climate Change factors:")
    for substance, value in cf.get_factors("climate_change").items():
        print(f"  {substance}: {value}")
    
    print("\nIonizing Radiation factors (important for PG):")
    for substance, value in cf.get_factors("ionizing_radiation").items():
        print(f"  {substance}: {value}")
