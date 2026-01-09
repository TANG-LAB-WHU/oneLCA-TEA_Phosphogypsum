"""
PG-LCA-TEA Framework

Main package for phosphogypsum life cycle assessment and techno-economic analysis.
"""

__version__ = "1.1.0"

from pgloop.lca.engine import LCAEngine
from pgloop.tea.engine import TEAEngine
from pgloop.pathways import get_pathway, list_pathways
from pgloop.risk.aggregator import RiskAggregator
from pgloop.decision.recommender import PathwayRanker
from pgloop.uncertainty.direct_sampling import MonteCarloSimulator

__all__ = [
    "LCAEngine",
    "TEAEngine",
    "get_pathway",
    "list_pathways",
    "RiskAggregator",
    "PathwayRanker",
    "MonteCarloSimulator",
]

