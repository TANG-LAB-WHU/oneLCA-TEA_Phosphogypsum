"""
Chemicals Module

Chemical species definitions, property database, and ML-based property prediction.
Integrates with MACE universal force field for missing property estimation.
"""

from pgloop.chemicals.base_chemical import Chemical, ChemicalConsumption
from pgloop.chemicals.registry import (
    CHEMICAL_DATABASE,
    get_chemical,
    list_chemicals,
)
from pgloop.chemicals.property_predictor import (
    PropertyPredictor,
    PropertyPrediction,
)
from pgloop.chemicals.acids import ACIDS
from pgloop.chemicals.bases import BASES
from pgloop.chemicals.solvents import SOLVENTS

__all__ = [
    "Chemical",
    "ChemicalConsumption",
    "CHEMICAL_DATABASE",
    "get_chemical",
    "list_chemicals",
    "PropertyPredictor",
    "PropertyPrediction",
    "ACIDS",
    "BASES",
    "SOLVENTS",
]
