"""
Shared data schemas for dynamic assessment workflows.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TimeSeriesPoint:
    """Single metric value at a given year."""

    year: int
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PosteriorSummary:
    """Posterior summary statistics for one metric."""

    mean: float
    std: float
    p5: float
    p50: float
    p95: float


@dataclass
class DynamicAssessmentResult:
    """
    Normalized result payload used by dynamic assessment pipeline.
    """

    pathway_code: str
    scenario_name: str
    time_series_metrics: List[TimeSeriesPoint] = field(default_factory=list)
    posterior_summary: Dict[str, PosteriorSummary] = field(default_factory=dict)
    robustness_stats: Dict[str, float] = field(default_factory=dict)
    sobol_indices: Dict[str, Dict[str, float]] = field(default_factory=dict)
    stochastic_density_summary: Dict[str, float] = field(default_factory=dict)
