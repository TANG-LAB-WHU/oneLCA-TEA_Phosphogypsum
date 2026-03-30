"""
Chemical Base Class

Core data structures for chemical species with property management.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Property units mapping
PROPERTY_UNITS = {
    "density": "kg/m3",
    "molecular_weight": "g/mol",
    "boiling_point": "K",
    "melting_point": "K",
    "heat_capacity": "J/(mol·K)",
    "viscosity": "Pa·s",
    "vapor_pressure": "Pa",
    "solubility": "g/L",
}

# Default values for missing properties
PROPERTY_DEFAULTS = {
    "density": 1000.0,
    "heat_capacity": 75.0,
    "viscosity": 0.001,
}


@dataclass
class Chemical:
    """
    Chemical species with comprehensive property data.

    Attributes:
        name: Common name
        formula: Molecular formula
        cas_number: CAS registry number
        smiles: SMILES representation for ML prediction
        molecular_weight: Molecular weight (g/mol)
        density_kg_m3: Density at 25°C
        state: Physical state (solid, liquid, gas)
        gwp_kg_co2_per_kg: Global warming potential from production
        price_usd_per_kg: Unit price
        hazard_class: GHS hazard classification
    """

    name: str
    formula: str
    cas_number: str = ""
    smiles: str = ""

    # Physical properties
    molecular_weight: float = 0.0
    density_kg_m3: float = 1000.0
    boiling_point_k: Optional[float] = None
    melting_point_k: Optional[float] = None
    heat_capacity_j_mol_k: Optional[float] = None
    state: str = "liquid"

    # LCA data (upstream production impacts)
    gwp_kg_co2_per_kg: float = 0.0
    acidification_kg_so2_per_kg: float = 0.0
    eutrophication_kg_po4_per_kg: float = 0.0

    # TEA data
    price_usd_per_kg: float = 0.0
    price_region: str = "US"
    price_year: int = 2024

    # Safety/compliance
    hazard_class: str = ""

    # Additional properties
    properties: Dict[str, Any] = field(default_factory=dict)

    # Property predictor (lazy loaded)
    _predictor: Any = field(default=None, repr=False)

    def get_property(self, prop_name: str, temperature_k: float = 298.15) -> float:
        """
        Get property value with automatic prediction fallback.

        Args:
            prop_name: Property name (density, heat_capacity, etc.)
            temperature_k: Temperature in Kelvin

        Returns:
            Property value
        """
        # Check if property exists directly
        attr_map = {
            "density": "density_kg_m3",
            "molecular_weight": "molecular_weight",
            "heat_capacity": "heat_capacity_j_mol_k",
            "boiling_point": "boiling_point_k",
            "melting_point": "melting_point_k",
        }

        if prop_name in attr_map:
            value = getattr(self, attr_map[prop_name], None)
            if value is not None:
                return value

        # Check additional properties
        if prop_name in self.properties:
            return self.properties[prop_name]

        # Use predictor if SMILES available
        if self.smiles:
            if self._predictor is None:
                from pgloop.chemicals.property_predictor import PropertyPredictor

                self._predictor = PropertyPredictor(use_mace=True)

            prediction = self._predictor.get_property(self.smiles, prop_name, temperature_k)
            return prediction.value

        # Return default
        return PROPERTY_DEFAULTS.get(prop_name, 0.0)

    def get_lci_impact(self, quantity_kg: float) -> Dict[str, float]:
        """
        Calculate upstream LCI impacts for given quantity.

        Args:
            quantity_kg: Amount of chemical used (kg)

        Returns:
            Dict of impact category -> value
        """
        return {
            "gwp_kg_co2": self.gwp_kg_co2_per_kg * quantity_kg,
            "acidification_kg_so2": self.acidification_kg_so2_per_kg * quantity_kg,
            "eutrophication_kg_po4": self.eutrophication_kg_po4_per_kg * quantity_kg,
        }

    def get_cost(self, quantity_kg: float) -> float:
        """Calculate chemical cost."""
        return self.price_usd_per_kg * quantity_kg

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "formula": self.formula,
            "cas_number": self.cas_number,
            "smiles": self.smiles,
            "molecular_weight": self.molecular_weight,
            "density_kg_m3": self.density_kg_m3,
            "state": self.state,
            "gwp_kg_co2_per_kg": self.gwp_kg_co2_per_kg,
            "price_usd_per_kg": self.price_usd_per_kg,
            "hazard_class": self.hazard_class,
        }


@dataclass
class ChemicalConsumption:
    """
    Chemical consumption model for a process.

    Links a chemical to its consumption rate in a pathway.
    """

    chemical: Chemical
    rate_kg_per_t_input: float  # kg chemical per tonne PG input
    purpose: str = ""  # e.g., "leaching agent", "pH control"
    recovery_rate: float = 0.0  # Fraction recoverable/recyclable

    def get_lci_contribution(self, throughput_kg: float) -> Dict:
        """
        Get LCI contribution for this chemical consumption.

        Args:
            throughput_kg: PG throughput (kg)

        Returns:
            Dict with 'inputs', 'impacts', 'cost'
        """
        consumption_kg = self.rate_kg_per_t_input * throughput_kg / 1000
        net_consumption = consumption_kg * (1 - self.recovery_rate)

        return {
            "inputs": {self.chemical.name: net_consumption},
            "impacts": self.chemical.get_lci_impact(net_consumption),
            "cost": self.chemical.get_cost(net_consumption),
        }

    def get_cost(self, throughput_kg: float) -> float:
        """Calculate chemical cost for given throughput."""
        consumption_kg = self.rate_kg_per_t_input * throughput_kg / 1000
        net_consumption = consumption_kg * (1 - self.recovery_rate)
        return self.chemical.get_cost(net_consumption)
