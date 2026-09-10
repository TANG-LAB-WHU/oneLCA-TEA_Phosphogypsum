"""
PG-LCA-TEA Framework

Main package for phosphogypsum life cycle assessment and techno-economic analysis.
"""

__version__ = "0.7.0"
from pgloop.assessment import IntegratedAssessmentEngine
from pgloop.decision.recommender import PathwayRanker
from pgloop.iodata.datahub import DataHub
from pgloop.lca.lca_engine import LCAEngine, LCAResult
from pgloop.pathways import get_pathway, list_pathways
from pgloop.risk.aggregator import RiskAggregator
from pgloop.tea.tea_engine import TEAEngine, TEAResult
from pgloop.uncertainty.direct_sampling import MonteCarloSimulator
from pgloop.uncertainty.propagation import JointUncertaintyPropagator

__all__ = [
    "DataHub",
    "IntegratedAssessmentEngine",
    "LCAEngine",
    "LCAResult",
    "TEAEngine",
    "TEAResult",
    "get_pathway",
    "list_pathways",
    "RiskAggregator",
    "PathwayRanker",
    "MonteCarloSimulator",
    "JointUncertaintyPropagator",
    "__version__",
]
