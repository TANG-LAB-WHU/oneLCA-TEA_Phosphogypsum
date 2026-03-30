"""
Chemical Reactors Module

Reactor equipment for chemical processing pathways.
"""

from typing import Dict

from pgloop.equipment.base_equipment import BaseEquipment


class CSTRReactor(BaseEquipment):
    """Continuous Stirred Tank Reactor."""

    def __init__(
        self,
        name: str = "CSTR",
        volume_m3: float = 10.0,
        material: str = "SS316",
        agitator_power_kw: float = 5.0,
        **kwargs,
    ):
        super().__init__(
            name=name,
            equipment_type="reactor",
            capacity=volume_m3,
            capacity_unit="m3",
            material=material,
            **kwargs,
        )
        self.volume_m3 = volume_m3
        self.agitator_power_kw = agitator_power_kw

    def get_base_cost(self, base_year: int = 2024) -> float:
        """Cost correlation for CSTR (Guthrie method)."""
        # Base cost = a * V^b, typical for jacketed vessels with agitator
        a, b = 15000, 0.6  # USD, exponent
        base = a * (self.volume_m3**b)

        # Pressure factor
        if self.pressure_bar > 5:
            base *= 1.0 + 0.1 * (self.pressure_bar - 5)

        return base

    def get_lci_data(self, throughput_kg: float) -> Dict:
        """LCI for reactor operation."""
        residence_time_h = self.properties.get("residence_time_h", 2.0)
        batches = throughput_kg / (self.volume_m3 * 1000)  # Assume density ~1000 kg/m3

        return {
            "inputs": {},
            "energy": {
                "electricity_kwh": self.agitator_power_kw * residence_time_h * batches,
                "heat_mj": self.properties.get("heat_duty_mj_per_t", 0) * throughput_kg / 1000,
            },
            "emissions": {},
        }

    def get_opex_data(self, throughput_kg: float, hours_per_year: float = 8000) -> Dict:
        return {
            "energy_kwh": self.agitator_power_kw * hours_per_year,
            "maintenance_usd": self.get_capex() * 0.03,  # 3% of CAPEX
            "labor_hours": hours_per_year * 0.1,  # 10% operator attention
        }


class BatchReactor(BaseEquipment):
    """Batch Reactor for discontinuous processes."""

    def __init__(
        self,
        name: str = "Batch Reactor",
        volume_m3: float = 5.0,
        material: str = "SS316",
        cycle_time_h: float = 4.0,
        **kwargs,
    ):
        super().__init__(
            name=name,
            equipment_type="reactor",
            capacity=volume_m3,
            capacity_unit="m3",
            material=material,
            **kwargs,
        )
        self.volume_m3 = volume_m3
        self.cycle_time_h = cycle_time_h

    def get_base_cost(self, base_year: int = 2024) -> float:
        a, b = 12000, 0.65
        return a * (self.volume_m3**b)

    def get_lci_data(self, throughput_kg: float) -> Dict:
        cycles = throughput_kg / (self.volume_m3 * 800)  # 80% fill, density 1000
        return {
            "inputs": {},
            "energy": {"electricity_kwh": 3 * self.cycle_time_h * cycles},
            "emissions": {},
        }

    def get_opex_data(self, throughput_kg: float, hours_per_year: float = 8000) -> Dict:
        return {
            "energy_kwh": 3 * hours_per_year,
            "maintenance_usd": self.get_capex() * 0.04,
            "labor_hours": hours_per_year * 0.15,
        }


class LeachingTank(BaseEquipment):
    """Acid Leaching Tank for REE extraction."""

    def __init__(
        self,
        name: str = "Leaching Tank",
        volume_m3: float = 20.0,
        material: str = "PVDF",  # Acid resistant
        acid_type: str = "H2SO4",
        **kwargs,
    ):
        super().__init__(
            name=name,
            equipment_type="reactor",
            capacity=volume_m3,
            capacity_unit="m3",
            material=material,
            **kwargs,
        )
        self.volume_m3 = volume_m3
        self.acid_type = acid_type

    def get_base_cost(self, base_year: int = 2024) -> float:
        # Lined tanks are more expensive
        a, b = 20000, 0.55
        return a * (self.volume_m3**b)

    def get_lci_data(self, throughput_kg: float) -> Dict:
        acid_consumption = self.properties.get("acid_kg_per_t", 300) * throughput_kg / 1000
        return {
            "inputs": {self.acid_type: acid_consumption},
            "energy": {"electricity_kwh": 5 * throughput_kg / 1000},
            "emissions": {"SO2_air": acid_consumption * 0.001},
        }

    def get_opex_data(self, throughput_kg: float, hours_per_year: float = 8000) -> Dict:
        return {
            "energy_kwh": 5 * hours_per_year,
            "maintenance_usd": self.get_capex() * 0.05,  # Higher for corrosive service
            "labor_hours": hours_per_year * 0.12,
        }


class MixingTank(BaseEquipment):
    """Simple mixing/holding tank with agitator."""

    def __init__(
        self, name: str = "Mixing Tank", volume_m3: float = 10.0, material: str = "SS304", **kwargs
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
        a, b = 8000, 0.6
        return a * (self.volume_m3**b)

    def get_lci_data(self, throughput_kg: float) -> Dict:
        return {
            "inputs": {},
            "energy": {"electricity_kwh": 2 * throughput_kg / 1000},
            "emissions": {},
        }

    def get_opex_data(self, throughput_kg: float, hours_per_year: float = 8000) -> Dict:
        return {
            "energy_kwh": 2 * hours_per_year,
            "maintenance_usd": self.get_capex() * 0.02,
            "labor_hours": hours_per_year * 0.05,
        }
