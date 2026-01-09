"""
Separation Equipment Module

Equipment for solid-liquid separation, evaporation, and extraction.
"""

from typing import Dict
from pgloop.equipment.base_equipment import BaseEquipment


class FilterPress(BaseEquipment):
    """Plate and Frame Filter Press for solid-liquid separation."""
    
    def __init__(
        self,
        name: str = "Filter Press",
        filter_area_m2: float = 20.0,
        material: str = "SS316",
        **kwargs
    ):
        super().__init__(
            name=name,
            equipment_type="filter",
            capacity=filter_area_m2,
            capacity_unit="m2",
            material=material,
            **kwargs
        )
        self.filter_area_m2 = filter_area_m2
    
    def get_base_cost(self, base_year: int = 2024) -> float:
        # Cost per m2 filter area
        a, b = 5000, 0.7
        return a * (self.filter_area_m2 ** b)
    
    def get_lci_data(self, throughput_kg: float) -> Dict:
        # Filter cloth replacement, wash water
        cycles = throughput_kg / (self.filter_area_m2 * 50)  # kg/m2 per cycle
        return {
            "inputs": {
                "wash_water_kg": cycles * 100,
                "filter_cloth_m2": self.filter_area_m2 * 0.001,  # Replacement rate
            },
            "energy": {"electricity_kwh": 0.5 * cycles},
            "emissions": {}
        }
    
    def get_opex_data(self, throughput_kg: float, hours_per_year: float = 8000) -> Dict:
        return {
            "energy_kwh": 50 * hours_per_year / 8000,  # Normalized
            "maintenance_usd": self.get_capex() * 0.08,  # High consumables
            "labor_hours": hours_per_year * 0.15,
        }


class Centrifuge(BaseEquipment):
    """Decanter Centrifuge for continuous solid-liquid separation."""
    
    def __init__(
        self,
        name: str = "Decanter Centrifuge",
        capacity_m3h: float = 5.0,
        material: str = "SS316",
        **kwargs
    ):
        super().__init__(
            name=name,
            equipment_type="filter",
            capacity=capacity_m3h,
            capacity_unit="m3/h",
            material=material,
            **kwargs
        )
        self.capacity_m3h = capacity_m3h
    
    def get_base_cost(self, base_year: int = 2024) -> float:
        a, b = 80000, 0.5
        return a * (self.capacity_m3h ** b)
    
    def get_lci_data(self, throughput_kg: float) -> Dict:
        operating_hours = throughput_kg / (self.capacity_m3h * 1000)
        return {
            "inputs": {},
            "energy": {"electricity_kwh": 15 * operating_hours},  # ~15 kW typical
            "emissions": {}
        }
    
    def get_opex_data(self, throughput_kg: float, hours_per_year: float = 8000) -> Dict:
        return {
            "energy_kwh": 15 * hours_per_year,
            "maintenance_usd": self.get_capex() * 0.06,
            "labor_hours": hours_per_year * 0.08,
        }


class Evaporator(BaseEquipment):
    """Single/Multi-effect Evaporator for concentration."""
    
    def __init__(
        self,
        name: str = "Evaporator",
        evaporation_rate_m3h: float = 2.0,
        effects: int = 3,
        material: str = "SS316",
        **kwargs
    ):
        super().__init__(
            name=name,
            equipment_type="heat_exchanger",
            capacity=evaporation_rate_m3h,
            capacity_unit="m3/h evap",
            material=material,
            **kwargs
        )
        self.evaporation_rate_m3h = evaporation_rate_m3h
        self.effects = effects
    
    def get_base_cost(self, base_year: int = 2024) -> float:
        # Multi-effect reduces steam but increases CAPEX
        base_per_effect = 50000 * (self.evaporation_rate_m3h ** 0.6)
        return base_per_effect * self.effects * 0.7  # Economy of scale
    
    def get_lci_data(self, throughput_kg: float) -> Dict:
        water_evaporated = throughput_kg * 0.3  # Assume 30% water removal
        steam_per_kg = 1.2 / self.effects  # Steam economy improves with effects
        return {
            "inputs": {"steam_kg": water_evaporated * steam_per_kg},
            "energy": {"electricity_kwh": self.evaporation_rate_m3h * 5},
            "emissions": {}
        }
    
    def get_opex_data(self, throughput_kg: float, hours_per_year: float = 8000) -> Dict:
        steam_cost = self.evaporation_rate_m3h * 1000 * (1.2 / self.effects) * 0.03 * hours_per_year
        return {
            "energy_kwh": self.evaporation_rate_m3h * 5 * hours_per_year,
            "steam_cost_usd": steam_cost,
            "maintenance_usd": self.get_capex() * 0.04,
            "labor_hours": hours_per_year * 0.1,
        }


class SolventExtractor(BaseEquipment):
    """Mixer-Settler or Pulsed Column for solvent extraction."""
    
    def __init__(
        self,
        name: str = "Solvent Extractor",
        stages: int = 4,
        throughput_m3h: float = 5.0,
        material: str = "SS316",
        **kwargs
    ):
        super().__init__(
            name=name,
            equipment_type="vessel",
            capacity=throughput_m3h,
            capacity_unit="m3/h",
            material=material,
            **kwargs
        )
        self.stages = stages
        self.throughput_m3h = throughput_m3h
    
    def get_base_cost(self, base_year: int = 2024) -> float:
        cost_per_stage = 30000 * (self.throughput_m3h ** 0.5)
        return cost_per_stage * self.stages
    
    def get_lci_data(self, throughput_kg: float) -> Dict:
        # Solvent losses and makeup
        solvent_loss_rate = 0.001  # 0.1% loss per pass
        return {
            "inputs": {
                "organic_solvent_kg": throughput_kg * solvent_loss_rate,
            },
            "energy": {"electricity_kwh": 2 * self.stages * throughput_kg / 1000},
            "emissions": {"VOC_air": throughput_kg * solvent_loss_rate * 0.01}
        }
    
    def get_opex_data(self, throughput_kg: float, hours_per_year: float = 8000) -> Dict:
        return {
            "energy_kwh": 2 * self.stages * hours_per_year,
            "solvent_makeup_usd": self.throughput_m3h * 1000 * 0.001 * 5 * hours_per_year,
            "maintenance_usd": self.get_capex() * 0.04,
            "labor_hours": hours_per_year * 0.12,
        }
