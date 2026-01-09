"""
Risk Aggregator

Combines micro and macro risk assessments into overall risk score.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np


class RiskLevel(Enum):
    """Risk level classification."""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class RiskScore:
    """Individual risk assessment result."""
    
    category: str
    subcategory: str
    score: float  # 0-100
    level: RiskLevel
    factors: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    mitigation: List[str] = field(default_factory=list)
    
    @classmethod
    def from_score(cls, category: str, subcategory: str, score: float, **kwargs) -> "RiskScore":
        """Create RiskScore with auto-classified level."""
        if score < 20:
            level = RiskLevel.VERY_LOW
        elif score < 40:
            level = RiskLevel.LOW
        elif score < 60:
            level = RiskLevel.MEDIUM
        elif score < 80:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.VERY_HIGH
        
        return cls(category=category, subcategory=subcategory, score=score, level=level, **kwargs)


@dataclass
class AggregatedRisk:
    """Aggregated risk assessment result."""
    
    overall_score: float
    overall_level: RiskLevel
    micro_score: float
    macro_score: float
    scores: List[RiskScore]
    risk_adjusted_discount_rate: float
    risk_premium: float
    
    def to_dict(self) -> Dict:
        return {
            "overall_score": self.overall_score,
            "overall_level": self.overall_level.name,
            "micro_score": self.micro_score,
            "macro_score": self.macro_score,
            "risk_adjusted_discount_rate": self.risk_adjusted_discount_rate,
            "risk_premium": self.risk_premium,
            "n_risk_factors": len(self.scores),
        }


class RiskAggregator:
    """
    Aggregates micro and macro risk scores into overall assessment.
    
    Uses weighted averaging with configurable weights.
    """
    
    # Default weights for risk categories
    DEFAULT_WEIGHTS = {
        # Micro risks
        "technical": 0.20,
        "operational": 0.15,
        "financial": 0.15,
        # Macro risks
        "political": 0.15,
        "economic": 0.15,
        "market": 0.10,
        "policy": 0.10,
    }
    
    def __init__(
        self,
        weights: Dict[str, float] = None,
        base_discount_rate: float = 0.08
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.base_discount_rate = base_discount_rate
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Ensure weights sum to 1."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def aggregate(self, risk_scores: List[RiskScore]) -> AggregatedRisk:
        """
        Aggregate multiple risk scores into overall assessment.
        
        Args:
            risk_scores: List of individual risk assessments
            
        Returns:
            AggregatedRisk with overall metrics
        """
        if not risk_scores:
            return AggregatedRisk(
                overall_score=0,
                overall_level=RiskLevel.VERY_LOW,
                micro_score=0,
                macro_score=0,
                scores=[],
                risk_adjusted_discount_rate=self.base_discount_rate,
                risk_premium=0,
            )
        
        # Separate micro and macro
        micro_scores = [s for s in risk_scores if s.category in ["technical", "operational", "financial"]]
        macro_scores = [s for s in risk_scores if s.category in ["political", "economic", "market", "policy"]]
        
        # Calculate weighted averages
        micro_avg = self._weighted_average(micro_scores)
        macro_avg = self._weighted_average(macro_scores)
        
        # Overall is weighted combination
        micro_weight = sum(self.weights.get(s.category, 0) for s in micro_scores)
        macro_weight = sum(self.weights.get(s.category, 0) for s in macro_scores)
        total_weight = micro_weight + macro_weight
        
        if total_weight > 0:
            overall = (micro_avg * micro_weight + macro_avg * macro_weight) / total_weight
        else:
            overall = (micro_avg + macro_avg) / 2
        
        # Risk premium and adjusted discount rate
        risk_premium = self._calculate_risk_premium(overall)
        adjusted_rate = self.base_discount_rate + risk_premium
        
        # Classification
        if overall < 20:
            level = RiskLevel.VERY_LOW
        elif overall < 40:
            level = RiskLevel.LOW
        elif overall < 60:
            level = RiskLevel.MEDIUM
        elif overall < 80:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.VERY_HIGH
        
        return AggregatedRisk(
            overall_score=overall,
            overall_level=level,
            micro_score=micro_avg,
            macro_score=macro_avg,
            scores=risk_scores,
            risk_adjusted_discount_rate=adjusted_rate,
            risk_premium=risk_premium,
        )
    
    def _weighted_average(self, scores: List[RiskScore]) -> float:
        """Calculate weighted average of scores."""
        if not scores:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for score in scores:
            weight = self.weights.get(score.category, 1.0 / len(scores))
            weighted_sum += score.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_risk_premium(self, overall_score: float) -> float:
        """
        Calculate risk premium for discount rate adjustment.
        
        Uses non-linear mapping: higher risk = disproportionately higher premium.
        """
        # Exponential mapping: 0-100 score -> 0-10% premium
        return 0.10 * (np.exp(overall_score / 100) - 1) / (np.e - 1)
    
    def calculate_risk_adjusted_npv(
        self,
        cash_flows: List[float],
        aggregated_risk: AggregatedRisk
    ) -> float:
        """
        Calculate risk-adjusted NPV using adjusted discount rate.
        
        Args:
            cash_flows: List of annual cash flows (year 0, 1, 2, ...)
            aggregated_risk: Aggregated risk assessment
            
        Returns:
            Risk-adjusted NPV
        """
        rate = aggregated_risk.risk_adjusted_discount_rate
        npv = 0.0
        
        for t, cf in enumerate(cash_flows):
            npv += cf / ((1 + rate) ** t)
        
        return npv
