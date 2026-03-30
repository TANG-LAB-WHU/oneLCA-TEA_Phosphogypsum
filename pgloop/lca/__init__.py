"""
LCA Module

Life Cycle Assessment calculation engine for phosphogypsum treatment.
Implements ISO 14040/14044 methodology with pure Python.
"""

from pgloop.lca.characterization import CharacterizationFactors
from pgloop.lca.impact_assessment import ImpactAssessment
from pgloop.lca.inventory import LifeCycleInventory
from pgloop.lca.lca_engine import LCAEngine

__all__ = ["LCAEngine", "LifeCycleInventory", "ImpactAssessment", "CharacterizationFactors"]
