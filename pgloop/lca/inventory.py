"""
Life Cycle Inventory Module

Handles LCI data structures and calculations.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Flow:
    """Represents a material or energy flow."""

    name: str
    quantity: float
    unit: str
    compartment: str = ""  # air, water, soil, etc.
    sub_compartment: str = ""
    uncertainty_min: Optional[float] = None
    uncertainty_max: Optional[float] = None
    source: str = ""

    def scale(self, factor: float) -> "Flow":
        """Return a scaled copy of this flow."""
        return Flow(
            name=self.name,
            quantity=self.quantity * factor,
            unit=self.unit,
            compartment=self.compartment,
            sub_compartment=self.sub_compartment,
            uncertainty_min=self.uncertainty_min * factor if self.uncertainty_min else None,
            uncertainty_max=self.uncertainty_max * factor if self.uncertainty_max else None,
            source=self.source,
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "quantity": self.quantity,
            "unit": self.unit,
            "compartment": self.compartment,
            "sub_compartment": self.sub_compartment,
            "uncertainty_min": self.uncertainty_min,
            "uncertainty_max": self.uncertainty_max,
            "source": self.source,
        }


@dataclass
class LifeCycleInventory:
    """
    Life Cycle Inventory for a process or pathway.

    Contains:
    - Inputs (materials, energy)
    - Outputs (products, co-products)
    - Emissions (to air, water, soil)
    - Avoided products (system expansion)
    """

    process_name: str
    functional_unit: str = "1 kg"
    functional_unit_value: float = 1.0

    inputs: List[Flow] = field(default_factory=list)
    outputs: List[Flow] = field(default_factory=list)
    emissions_air: List[Flow] = field(default_factory=list)
    emissions_water: List[Flow] = field(default_factory=list)
    emissions_soil: List[Flow] = field(default_factory=list)
    avoided_products: List[Flow] = field(default_factory=list)

    metadata: Dict = field(default_factory=dict)

    def add_input(self, name: str, quantity: float, unit: str, **kwargs) -> None:
        """Add an input flow."""
        self.inputs.append(Flow(name=name, quantity=quantity, unit=unit, **kwargs))

    def add_output(self, name: str, quantity: float, unit: str, **kwargs) -> None:
        """Add an output flow."""
        self.outputs.append(Flow(name=name, quantity=quantity, unit=unit, **kwargs))

    def add_emission(
        self, name: str, quantity: float, unit: str, compartment: str, **kwargs
    ) -> None:
        """Add an emission flow."""
        flow = Flow(name=name, quantity=quantity, unit=unit, compartment=compartment, **kwargs)

        if compartment == "air":
            self.emissions_air.append(flow)
        elif compartment == "water":
            self.emissions_water.append(flow)
        elif compartment == "soil":
            self.emissions_soil.append(flow)
        else:
            raise ValueError(f"Unknown compartment: {compartment}")

    def add_avoided_product(self, name: str, quantity: float, unit: str, **kwargs) -> None:
        """Add an avoided product (system expansion)."""
        self.avoided_products.append(Flow(name=name, quantity=quantity, unit=unit, **kwargs))

    def scale_to(self, target_value: float) -> "LifeCycleInventory":
        """
        Scale all flows to a new functional unit value.

        Args:
            target_value: New functional unit value

        Returns:
            New LifeCycleInventory with scaled flows
        """
        factor = target_value / self.functional_unit_value

        scaled = LifeCycleInventory(
            process_name=self.process_name,
            functional_unit=self.functional_unit,
            functional_unit_value=target_value,
            metadata=copy.deepcopy(self.metadata),
        )

        scaled.inputs = [f.scale(factor) for f in self.inputs]
        scaled.outputs = [f.scale(factor) for f in self.outputs]
        scaled.emissions_air = [f.scale(factor) for f in self.emissions_air]
        scaled.emissions_water = [f.scale(factor) for f in self.emissions_water]
        scaled.emissions_soil = [f.scale(factor) for f in self.emissions_soil]
        scaled.avoided_products = [f.scale(factor) for f in self.avoided_products]

        return scaled

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "process_name": self.process_name,
            "functional_unit": self.functional_unit,
            "functional_unit_value": self.functional_unit_value,
            "inputs": [f.to_dict() for f in self.inputs],
            "outputs": [f.to_dict() for f in self.outputs],
            "emissions_air": [f.to_dict() for f in self.emissions_air],
            "emissions_water": [f.to_dict() for f in self.emissions_water],
            "emissions_soil": [f.to_dict() for f in self.emissions_soil],
            "avoided_products": [f.to_dict() for f in self.avoided_products],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LifeCycleInventory":
        """Create from dictionary."""
        lci = cls(
            process_name=data.get("process_name", "Unknown"),
            functional_unit=data.get("functional_unit", "1 kg"),
            functional_unit_value=data.get("functional_unit_value", 1.0),
            metadata=data.get("metadata", {}),
        )

        for inp in data.get("inputs", []):
            lci.inputs.append(Flow(**inp))

        for out in data.get("outputs", []):
            lci.outputs.append(Flow(**out))

        for em in data.get("emissions_air", []):
            lci.emissions_air.append(Flow(**em))

        for em in data.get("emissions_water", []):
            lci.emissions_water.append(Flow(**em))

        for em in data.get("emissions_soil", []):
            lci.emissions_soil.append(Flow(**em))

        for ap in data.get("avoided_products", []):
            lci.avoided_products.append(Flow(**ap))

        return lci

    def merge(self, other: "LifeCycleInventory") -> "LifeCycleInventory":
        """
        Merge with another LCI (e.g., for multi-stage processes).

        Args:
            other: Another LifeCycleInventory to merge

        Returns:
            New merged LifeCycleInventory
        """
        merged = LifeCycleInventory(
            process_name=f"{self.process_name} + {other.process_name}",
            functional_unit=self.functional_unit,
            functional_unit_value=self.functional_unit_value,
        )

        # Combine all flows
        merged.inputs = self.inputs + other.inputs
        merged.outputs = self.outputs + other.outputs
        merged.emissions_air = self.emissions_air + other.emissions_air
        merged.emissions_water = self.emissions_water + other.emissions_water
        merged.emissions_soil = self.emissions_soil + other.emissions_soil
        merged.avoided_products = self.avoided_products + other.avoided_products

        return merged

    def get_mass_balance(self) -> Dict[str, float]:
        """Calculate mass balance (inputs vs outputs + emissions)."""
        # Sum inputs (kg only)
        total_inputs = sum(f.quantity for f in self.inputs if f.unit == "kg")

        # Sum outputs and emissions
        total_outputs = sum(f.quantity for f in self.outputs if f.unit == "kg")

        total_emissions = sum(
            f.quantity
            for f in self.emissions_air + self.emissions_water + self.emissions_soil
            if f.unit == "kg"
        )

        return {
            "inputs_kg": total_inputs,
            "outputs_kg": total_outputs,
            "emissions_kg": total_emissions,
            "balance": total_inputs - total_outputs - total_emissions,
            "balance_pct": (
                ((total_inputs - total_outputs - total_emissions) / total_inputs * 100)
                if total_inputs > 0
                else 0
            ),
        }


def main():
    # Example usage
    lci = LifeCycleInventory(
        process_name="PG Stack Disposal", functional_unit="1 tonne PG", functional_unit_value=1000
    )

    # Add inputs
    lci.add_input("Phosphogypsum", 1000, "kg")
    lci.add_input("Diesel", 5, "kg")
    lci.add_input("Electricity", 10, "kWh")

    # Add emissions
    lci.add_emission("CO2", 50, "kg", "air")
    lci.add_emission("Radon-222", 1000, "Bq", "air")
    lci.add_emission("Fluoride", 0.1, "kg", "water")

    print(lci.to_dict())
    print(lci.get_mass_balance())


if __name__ == "__main__":
    main()
