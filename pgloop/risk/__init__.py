"""
Risk Assessment Module

Modular risk assessment combining micro-level (project/technology)
and macro-level (country/market/policy) risk factors.
"""

from pgloop.risk.aggregator import RiskAggregator, RiskScore
from pgloop.risk.macro.economic import EconomicRisk
from pgloop.risk.macro.market import MarketRisk
from pgloop.risk.macro.policy import PolicyRisk
from pgloop.risk.macro.political import PoliticalRisk
from pgloop.risk.micro.financial import ProjectFinancialRisk
from pgloop.risk.micro.operational import OperationalRisk
from pgloop.risk.micro.technical import TechnicalRisk

__all__ = [
    # Micro risks
    "TechnicalRisk",
    "OperationalRisk",
    "ProjectFinancialRisk",
    # Macro risks
    "PoliticalRisk",
    "EconomicRisk",
    "MarketRisk",
    "PolicyRisk",
    # Aggregation
    "RiskAggregator",
    "RiskScore",
]
