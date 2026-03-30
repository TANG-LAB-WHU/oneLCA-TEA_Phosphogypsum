"""
Material Handling Equipment Module

Equipment for crushing, drying, conveying, and storage.
"""

from typing import Dict

from pgloop.equipment.base_equipment import BaseEquipment


class Crusher(BaseEquipment):
    """Jaw/Impact Crusher for size reduction."""

    def __init__(
        self,
        name: str = "Crusher",
        capacity_tph: float = 10.0,
        power_kw: float = 50.0,
        material: str = "CS",
        **kwargs,
    ):
        super().__init__(
            name=name,
            equipment_type="crusher",
            capacity=capacity_tph,
            capacity_unit="t/h",
            material=material,
            **kwargs,
        )
        self.capacity_tph = capacity_tph
        self.power_kw = power_kw

    def get_base_cost(self, base_year: int = 2024) -> float:
        a, b = 25000, 0.7
        return a * (self.capacity_tph**b)

    def get_lci_data(self, throughput_kg: float) -> Dict:
        operating_hours = throughput_kg / (self.capacity_tph * 1000)
        return {
            "inputs": {},
            "energy": {"electricity_kwh": self.power_kw * operating_hours},
            "emissions": {"dust_air": throughput_kg * 0.0001},  # 0.01% as dust
        }

    def get_opex_data(self, throughput_kg: float, hours_per_year: float = 8000) -> Dict:
        return {
            "energy_kwh": self.power_kw * hours_per_year,
            "maintenance_usd": self.get_capex() * 0.10,  # High wear parts
            "labor_hours": hours_per_year * 0.05,
        }


class Dryer(BaseEquipment):
    """Rotary or Flash Dryer for moisture removal."""

    def __init__(
        self,
        name: str = "Rotary Dryer",
        evaporation_capacity_tph: float = 1.0,
        heat_source: str = "natural_gas",
        material: str = "CS",
        **kwargs,
    ):
        super().__init__(
            name=name,
            equipment_type="dryer",
            capacity=evaporation_capacity_tph,
            capacity_unit="t water/h",
            material=material,
            **kwargs,
        )
        self.evaporation_capacity_tph = evaporation_capacity_tph
        self.heat_source = heat_source

    def get_base_cost(self, base_year: int = 2024) -> float:
        a, b = 150000, 0.55
        return a * (self.evaporation_capacity_tph**b)

    def get_lci_data(self, throughput_kg: float) -> Dict:
        moisture_content = self.properties.get("moisture_in", 0.20)
        target_moisture = self.properties.get("moisture_out", 0.05)
        water_evaporated = throughput_kg * (moisture_content - target_moisture)

        # Heat requirement: ~2.5 MJ/kg water evaporated
        heat_required_mj = water_evaporated * 2.5

        emissions = {}
        if self.heat_source == "natural_gas":
            emissions["CO2_air"] = heat_required_mj * 0.055  # kg CO2/MJ

        return {
            "inputs": {"natural_gas_mj": heat_required_mj},
            "energy": {"electricity_kwh": 15 * water_evaporated / 1000},
            "emissions": emissions,
        }

    def get_opex_data(self, throughput_kg: float, hours_per_year: float = 8000) -> Dict:
        return {
            "energy_kwh": 15 * self.evaporation_capacity_tph * hours_per_year,
            "fuel_mj": self.evaporation_capacity_tph * 2.5 * 1000 * hours_per_year,
            "maintenance_usd": self.get_capex() * 0.05,
            "labor_hours": hours_per_year * 0.08,
        }


class Conveyor(BaseEquipment):
    """Belt Conveyor for material transport."""

    def __init__(
        self,
        name: str = "Belt Conveyor",
        length_m: float = 50.0,
        capacity_tph: float = 20.0,
        material: str = "CS",
        **kwargs,
    ):
        super().__init__(
            name=name,
            equipment_type="conveyor",
            capacity=capacity_tph,
            capacity_unit="t/h",
            material=material,
            **kwargs,
        )
        self.length_m = length_m
        self.capacity_tph = capacity_tph

    def get_base_cost(self, base_year: int = 2024) -> float:
        # Cost per meter increases with capacity
        cost_per_m = 500 * (self.capacity_tph**0.3)
        return cost_per_m * self.length_m

    def get_lci_data(self, throughput_kg: float) -> Dict:
        # Power: ~0.05 kW per m for typical belt
        power_kw = 0.05 * self.length_m + 2  # +2 for drive
        operating_hours = throughput_kg / (self.capacity_tph * 1000)
        return {
            "inputs": {},
            "energy": {"electricity_kwh": power_kw * operating_hours},
            "emissions": {},
        }

    def get_opex_data(self, throughput_kg: float, hours_per_year: float = 8000) -> Dict:
        power_kw = 0.05 * self.length_m + 2
        return {
            "energy_kwh": power_kw * hours_per_year,
            "maintenance_usd": self.get_capex() * 0.04,
            "labor_hours": hours_per_year * 0.02,
        }


class StorageSilo(BaseEquipment):
    """Bulk storage silo for solids."""

    def __init__(
        self, name: str = "Storage Silo", volume_m3: float = 100.0, material: str = "CS", **kwargs
    ):
        super().__init__(
            name=name,
            equipment_type="vessel",
            capacity=volume_m3,
            capacity_unit="m3",
            material=material,
            **kwargs,
        )
        self.volume_m3 = volume_m3

    def get_base_cost(self, base_year: int = 2024) -> float:
        a, b = 3000, 0.65
        return a * (self.volume_m3**b)

    def get_lci_data(self, throughput_kg: float) -> Dict:
        return {
            "inputs": {},
            "energy": {"electricity_kwh": 0.1 * throughput_kg / 1000},  # Minimal
            "emissions": {},
        }

    def get_opex_data(self, throughput_kg: float, hours_per_year: float = 8000) -> Dict:
        return {
            "energy_kwh": 10 * hours_per_year / 8000,
            "maintenance_usd": self.get_capex() * 0.01,
            "labor_hours": hours_per_year * 0.01,
        }
