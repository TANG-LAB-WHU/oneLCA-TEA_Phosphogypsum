"""
TEA Module

Techno-Economic Analysis calculation engine for phosphogypsum treatment.
Implements CLCC (Conventional) and SLCC (Societal) cost methodologies.
"""

from pgloop.tea.capex import CAPEXCalculator
from pgloop.tea.external_cost import ExternalCostCalculator
from pgloop.tea.opex import OPEXCalculator
from pgloop.tea.tea_engine import TEAEngine

__all__ = ["TEAEngine", "CAPEXCalculator", "OPEXCalculator", "ExternalCostCalculator"]
