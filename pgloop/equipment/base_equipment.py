"""
Base Equipment Class

Abstract base class for all chemical process equipment.
Provides interfaces for LCI, CAPEX, and OPEX calculations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

# Material factors for cost adjustment (relative to carbon steel)
MATERIAL_FACTORS = {
    "CS": 1.0,  # Carbon Steel
    "SS304": 1.3,  # Stainless Steel 304
    "SS316": 1.5,  # Stainless Steel 316
    "SS316L": 1.7,  # Stainless Steel 316L
    "Hastelloy": 2.5,  # Hastelloy C
    "Titanium": 3.5,  # Titanium
    "FRP": 0.8,  # Fiberglass Reinforced Plastic
    "PVDF": 1.2,  # Polyvinylidene fluoride lined
}

# Installation factors by equipment type (Lang factors)
INSTALLATION_FACTORS = {
    "reactor": 1.5,
    "vessel": 1.4,
    "heat_exchanger": 1.3,
    "pump": 1.4,
    "conveyor": 1.2,
    "dryer": 1.5,
    "filter": 1.4,
    "crusher": 1.3,
}


@dataclass
class EquipmentSpec:
    """Equipment specification data class."""

    name: str
    equipment_type: str
    capacity: float  # Primary capacity metric
    capacity_unit: str
    material: str = "SS316"
    operating_pressure_bar: float = 1.0
    operating_temperature_c: float = 25.0
    properties: Dict[str, Any] = field(default_factory=dict)


class BaseEquipment(ABC):
    """Abstract base class for process equipment."""

    def __init__(
        self,
        name: str,
        equipment_type: str,
        capacity: float,
        capacity_unit: str,
        material: str = "SS316",
        **kwargs,
    ):
        """
        Initialize equipment.

        Args:
            name: Equipment name/tag
            equipment_type: Type category (reactor, vessel, etc.)
            capacity: Primary capacity metric
            capacity_unit: Unit for capacity (m3, m2, kW, etc.)
            material: Construction material
        """
        self.name = name
        self.equipment_type = equipment_type
        self.capacity = capacity
        self.capacity_unit = capacity_unit
        self.material = material
        self.properties = kwargs

        # Operating conditions
        self.temperature_c = kwargs.get("temperature_c", 25.0)
        self.pressure_bar = kwargs.get("pressure_bar", 1.0)

    @property
    def material_factor(self) -> float:
        """Get material cost adjustment factor."""
        return MATERIAL_FACTORS.get(self.material, 1.5)

    @property
    def installation_factor(self) -> float:
        """Get installation cost factor (includes piping, electrical, etc.)."""
        return INSTALLATION_FACTORS.get(self.equipment_type, 1.4)

    # ==================== Abstract Methods ====================

    @abstractmethod
    def get_base_cost(self, base_year: int = 2024) -> float:
        """
        Calculate base equipment purchase cost (no factors applied).

        Args:
            base_year: Cost basis year

        Returns:
            Base equipment cost in USD
        """
        pass

    @abstractmethod
    def get_lci_data(self, throughput_kg: float) -> Dict:
        """
        Get Life Cycle Inventory data for this equipment.

        Args:
            throughput_kg: Material throughput in kg

        Returns:
            Dict with 'inputs', 'emissions', 'energy' keys
        """
        pass

    @abstractmethod
    def get_opex_data(self, throughput_kg: float, hours_per_year: float = 8000) -> Dict:
        """
        Get operational expenditure data.

        Args:
            throughput_kg: Material throughput in kg
            hours_per_year: Annual operating hours

        Returns:
            Dict with 'energy', 'maintenance', 'labor' keys
        """
        pass

    # ==================== Concrete Methods ====================

    def get_capex(self, include_installation: bool = True) -> float:
        """
        Calculate total CAPEX including material and installation factors.

        Args:
            include_installation: Whether to include installation costs

        Returns:
            Total CAPEX in USD
        """
        base_cost = self.get_base_cost()
        cost_with_material = base_cost * self.material_factor

        if include_installation:
            return cost_with_material * self.installation_factor
        return cost_with_material

    def get_capex_item(self) -> Dict:
        """Return CAPEX as a structured dict for pathway integration."""
        return {
            "name": self.name,
            "type": self.equipment_type,
            "base_cost": self.get_base_cost(),
            "material_factor": self.material_factor,
            "installation_factor": self.installation_factor,
            "total_cost": self.get_capex(),
        }

    def scale_cost(self, target_capacity: float, exponent: float = 0.6) -> float:
        """
        Scale equipment cost using power law (six-tenths rule).

        Args:
            target_capacity: New capacity
            exponent: Scaling exponent (typically 0.6-0.8)

        Returns:
            Scaled cost
        """
        if self.capacity <= 0:
            return self.get_base_cost()

        ratio = target_capacity / self.capacity
        return self.get_base_cost() * (ratio**exponent)

    def to_dict(self) -> Dict:
        """Serialize equipment to dictionary."""
        return {
            "name": self.name,
            "equipment_type": self.equipment_type,
            "capacity": self.capacity,
            "capacity_unit": self.capacity_unit,
            "material": self.material,
            "temperature_c": self.temperature_c,
            "pressure_bar": self.pressure_bar,
            "capex": self.get_capex(),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, {self.capacity} {self.capacity_unit})"
