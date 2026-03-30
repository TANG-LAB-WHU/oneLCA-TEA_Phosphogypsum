"""
Decision Support Module

Multi-criteria decision analysis for pathway selection.
Integrates LCA, TEA, and risk assessments.
"""

from pgloop.decision.criteria import (
    CriteriaSet,
    Criterion,
    create_default_criteria,
)
from pgloop.decision.mcda import (
    AHP,
    TOPSIS,
    WeightedSum,
)
from pgloop.decision.pareto import (
    ParetoAnalyzer,
    ParetoSolution,
)
from pgloop.decision.recommender import (
    PathwayRanker,
    Recommendation,
)
from pgloop.decision.dynamic_optimizer import DynamicMultiObjectiveOptimizer
from pgloop.decision.scenario import (
    DynamicScenarioAnalyzer,
    DynamicYearResult,
    Scenario,
    ScenarioAnalyzer,
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
    "DynamicScenarioAnalyzer",
    "DynamicYearResult",
    "DynamicMultiObjectiveOptimizer",
    "PathwayRanker",
    "Recommendation",
]
