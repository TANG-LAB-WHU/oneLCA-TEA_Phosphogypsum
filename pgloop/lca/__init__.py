"""
LCA Module

Life Cycle Assessment calculation engine for phosphogypsum treatment.
Implements ISO 14040/14044 methodology with pure Python.
"""

from pgloop.lca.engine import LCAEngine
from pgloop.lca.inventory import LifeCycleInventory
from pgloop.lca.impact_assessment import ImpactAssessment
from pgloop.lca.characterization import CharacterizationFactors

__all__ = ["LCAEngine", "LifeCycleInventory", "ImpactAssessment", "CharacterizationFactors"]
