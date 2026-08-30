"""
Chemicals Module

Chemical species definitions, property database, and ML-based property prediction.
Integrates with MACE universal force field for missing property estimation.
"""

from pgloop.chemicals.acids import ACIDS
from pgloop.chemicals.base_chemical import Chemical, ChemicalConsumption
from pgloop.chemicals.bases import BASES
from pgloop.chemicals.property_predictor import (
    PropertyPrediction,
    PropertyPredictor,
)
from pgloop.chemicals.registry import (
    CHEMICAL_DATABASE,
    get_chemical,
    list_chemicals,
)
from pgloop.chemicals.solvents import SOLVENTS
try:
    from pgloop.chemicals.mace_interface import get_mace_calculator, is_mace_available
    from pgloop.chemicals.lattice_optimizer import optimize_structure, fit_eos
    from pgloop.chemicals.eval_mace import fetch_mp_data, evaluate_mace_on_mp
except ImportError:
    get_mace_calculator = None
    is_mace_available = lambda: False
    optimize_structure = None
    fit_eos = None
    fetch_mp_data = None
    evaluate_mace_on_mp = None

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
    "get_mace_calculator",
    "is_mace_available",
    "optimize_structure",
    "fit_eos",
    "fetch_mp_data",
    "evaluate_mace_on_mp",
]
