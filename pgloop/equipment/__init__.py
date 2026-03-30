"""
Equipment / Unit Operations Module

Modular equipment components for LCA-TEA calculations.
Equipment classes provide standardized interfaces for LCI, CAPEX, and OPEX.
"""

from pgloop.equipment.base_equipment import BaseEquipment
from pgloop.equipment.heat_exchange import (
    CoolingTower,
    ShellTubeExchanger,
)
from pgloop.equipment.material_handling import (
    Conveyor,
    Crusher,
    Dryer,
    StorageSilo,
)
from pgloop.equipment.reactors import (
    BatchReactor,
    CSTRReactor,
    LeachingTank,
    MixingTank,
)
from pgloop.equipment.separations import (
    Centrifuge,
    Evaporator,
    FilterPress,
    SolventExtractor,
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
