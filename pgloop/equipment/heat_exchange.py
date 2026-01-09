"""
Heat Exchange Equipment Module

Heat exchangers, cooling towers, and thermal equipment.
"""

from typing import Dict
from pgloop.equipment.base_equipment import BaseEquipment


class ShellTubeExchanger(BaseEquipment):
    """Shell and Tube Heat Exchanger."""
    
    def __init__(
        self,
        name: str = "Shell & Tube HX",
        area_m2: float = 20.0,
        duty_kw: float = 100.0,
        material: str = "SS316",
        **kwargs
    ):
        super().__init__(
            name=name,
            equipment_type="heat_exchanger",
            capacity=area_m2,
            capacity_unit="m2",
            material=material,
            **kwargs
        )
        self.area_m2 = area_m2
        self.duty_kw = duty_kw
    
    def get_base_cost(self, base_year: int = 2024) -> float:
        # Cost correlation: area-based
        a, b = 3000, 0.65
        base = a * (self.area_m2 ** b)
        
        # Pressure adjustment
        if self.pressure_bar > 10:
            base *= 1.0 + 0.05 * (self.pressure_bar - 10)
        
        return base
    
    def get_lci_data(self, throughput_kg: float) -> Dict:
        # Heat exchangers are passive - energy from utilities
        hours = throughput_kg / 1000  # Simplified operating hours
        return {
            "inputs": {},
            "energy": {
                "cooling_water_m3": self.duty_kw * hours * 0.001 if self.properties.get("is_cooler") else 0,
                "steam_kg": self.duty_kw * hours * 0.5 if self.properties.get("is_heater") else 0,
            },
            "emissions": {}
        }
    
    def get_opex_data(self, throughput_kg: float, hours_per_year: float = 8000) -> Dict:
        return {
            "cooling_water_m3": self.duty_kw * hours_per_year * 0.001 if self.properties.get("is_cooler") else 0,
            "maintenance_usd": self.get_capex() * 0.03,
            "labor_hours": hours_per_year * 0.02,
        }


class CoolingTower(BaseEquipment):
    """Induced Draft Cooling Tower."""
    
    def __init__(
        self,
        name: str = "Cooling Tower",
        capacity_kw: float = 500.0,
        material: str = "FRP",
        **kwargs
    ):
        super().__init__(
            name=name,
            equipment_type="heat_exchanger",
            capacity=capacity_kw,
            capacity_unit="kW cooling",
            material=material,
            **kwargs
        )
        self.capacity_kw = capacity_kw
    
    def get_base_cost(self, base_year: int = 2024) -> float:
        a, b = 200, 0.7
        return a * (self.capacity_kw ** b)
    
    def get_lci_data(self, throughput_kg: float) -> Dict:
        hours = throughput_kg / 1000
        # Fan power: ~0.02 kW per kW cooling
        # Makeup water: ~3% of circulation rate
        return {
            "inputs": {
                "makeup_water_m3": self.capacity_kw * 0.0001 * hours,
            },
            "energy": {"electricity_kwh": self.capacity_kw * 0.02 * hours},
            "emissions": {}
        }
    
    def get_opex_data(self, throughput_kg: float, hours_per_year: float = 8000) -> Dict:
        return {
            "energy_kwh": self.capacity_kw * 0.02 * hours_per_year,
            "water_m3": self.capacity_kw * 0.0001 * hours_per_year,
            "maintenance_usd": self.get_capex() * 0.04,
            "labor_hours": hours_per_year * 0.03,
        }
