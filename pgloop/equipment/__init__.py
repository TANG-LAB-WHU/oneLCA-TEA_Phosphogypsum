"""
Equipment / Unit Operations Module

Modular equipment components for LCA-TEA calculations.
Equipment classes provide standardized interfaces for LCI, CAPEX, and OPEX.
"""

from pgloop.equipment.base_equipment import BaseEquipment
from pgloop.equipment.reactors import (
    CSTRReactor,
    BatchReactor,
    LeachingTank,
    MixingTank,
)
from pgloop.equipment.separations import (
    FilterPress,
    Centrifuge,
    Evaporator,
    SolventExtractor,
)
from pgloop.equipment.material_handling import (
    Crusher,
    Dryer,
    Conveyor,
    StorageSilo,
)
from pgloop.equipment.heat_exchange import (
    ShellTubeExchanger,
    CoolingTower,
)

__all__ = [
    "BaseEquipment",
    "CSTRReactor",
    "BatchReactor",
    "LeachingTank",
    "MixingTank",
    "FilterPress",
    "Centrifuge",
    "Evaporator",
    "SolventExtractor",
    "Crusher",
    "Dryer",
    "Conveyor",
    "StorageSilo",
    "ShellTubeExchanger",
    "CoolingTower",
]
