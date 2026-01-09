"""
Decision Support Module

Multi-criteria decision analysis for pathway selection.
Integrates LCA, TEA, and risk assessments.
"""

from pgloop.decision.criteria import (
    Criterion,
    CriteriaSet,
    create_default_criteria,
)
from pgloop.decision.mcda import (
    TOPSIS,
    AHP,
    WeightedSum,
)
from pgloop.decision.pareto import (
    ParetoAnalyzer,
    ParetoSolution,
)
from pgloop.decision.scenario import (
    Scenario,
    ScenarioAnalyzer,
)
from pgloop.decision.recommender import (
    PathwayRanker,
    Recommendation,
)

__all__ = [
    "Criterion",
    "CriteriaSet",
    "create_default_criteria",
    "TOPSIS",
    "AHP",
    "WeightedSum",
    "ParetoAnalyzer",
    "ParetoSolution",
    "Scenario",
    "ScenarioAnalyzer",
    "PathwayRanker",
    "Recommendation",
]
