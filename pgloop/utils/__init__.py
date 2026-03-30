"""
Utils Module

Common utilities for currency conversion, unit conversion,
annotations, and physical constants.
"""

from pgloop.utils.annotations import (
    Annotation,
    Assumption,
    DataSource,
    annotate,
)
from pgloop.utils.constants import (
    ATOMIC_MASSES,
    LCA_REFERENCE_VALUES,
    PHYSICAL_CONSTANTS,
)
from pgloop.utils.currency import (
    CurrencyConverter,
    adjust_inflation,
    convert_currency,
    get_exchange_rate,
)
from pgloop.utils.schema import (
    DynamicAssessmentResult,
    PosteriorSummary,
    TimeSeriesPoint,
)
from pgloop.utils.units import (
    UnitConverter,
    convert_energy,
    convert_mass,
    convert_temperature,
    convert_volume,
)

__all__ = [
    "convert_currency",
    "adjust_inflation",
    "get_exchange_rate",
    "CurrencyConverter",
    "convert_mass",
    "convert_energy",
    "convert_volume",
    "convert_temperature",
    "UnitConverter",
    "DataSource",
    "Assumption",
    "Annotation",
    "annotate",
    "PHYSICAL_CONSTANTS",
    "ATOMIC_MASSES",
    "LCA_REFERENCE_VALUES",
    "TimeSeriesPoint",
    "PosteriorSummary",
    "DynamicAssessmentResult",
]
