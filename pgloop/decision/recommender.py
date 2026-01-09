"""
Pathway Recommender Module

Integrates all analyses to provide ranked recommendations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pgloop.decision.criteria import CriteriaSet, create_default_criteria
from pgloop.decision.mcda import TOPSIS, WeightedSum, MCDAResult
from pgloop.decision.pareto import ParetoAnalyzer
from pgloop.decision.scenario import ScenarioAnalyzer


@dataclass
class Recommendation:
    """
    A pathway recommendation with explanation.
    """
    
    pathway_name: str
    rank: int
    score: float
    is_pareto_optimal: bool
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    explanation: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "pathway": self.pathway_name,
            "rank": self.rank,
            "score": self.score,
            "pareto_optimal": self.is_pareto_optimal,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "risks": self.risks,
            "explanation": self.explanation,
        }


class PathwayRanker:
    """
    Integrated pathway ranking and recommendation engine.
    """
    
    def __init__(
        self,
        criteria: CriteriaSet = None,
        method: str = "TOPSIS",
        lca_weight: float = 0.30,
        tea_weight: float = 0.40,
        risk_weight: float = 0.30,
    ):
        """
        Initialize ranker.
        
        Args:
            criteria: Custom criteria set (default: create_default_criteria)
            method: MCDA method ("TOPSIS", "WeightedSum", "AHP")
            lca_weight: Weight for environmental criteria
            tea_weight: Weight for economic criteria
            risk_weight: Weight for risk criteria
        """
        self.criteria = criteria or create_default_criteria()
        self.method = method
        
        # Store category weights for explanation
        self.category_weights = {
            "environmental": lca_weight,
            "economic": tea_weight,
            "risk": risk_weight,
        }
        
        # Adjust criteria weights by category
        self._adjust_category_weights(lca_weight, tea_weight, risk_weight)
    
    def _adjust_category_weights(
        self,
        lca_weight: float,
        tea_weight: float,
        risk_weight: float
    ) -> None:
        """Adjust individual criterion weights by category."""
        from pgloop.decision.criteria import Category
        
        # Calculate current category totals
        category_totals = {}
        for c in self.criteria.criteria:
            cat = c.category.value
            category_totals[cat] = category_totals.get(cat, 0) + c.weight
        
        # Rescale
        target_weights = {
            "environmental": lca_weight,
            "economic": tea_weight,
            "risk": risk_weight,
            "technical": 1 - lca_weight - tea_weight - risk_weight,
            "social": 0,
        }
        
        for c in self.criteria.criteria:
            cat = c.category.value
            if category_totals.get(cat, 0) > 0:
                c.weight = c.weight / category_totals[cat] * target_weights.get(cat, 0)
    
    def rank(
        self,
        pathways: Dict[str, Dict[str, float]]
    ) -> List[Recommendation]:
        """
        Rank pathways and generate recommendations.
        
        Args:
            pathways: {pathway_name: {criterion_name: value}}
            
        Returns:
            List of Recommendation objects in ranked order
        """
        # Select MCDA method
        if self.method == "TOPSIS":
            mcda = TOPSIS(self.criteria)
        else:
            mcda = WeightedSum(self.criteria)
        
        # Get rankings
        mcda_result = mcda.rank(pathways)
        
        # Pareto analysis
        pareto = ParetoAnalyzer()
        pareto_optimal = set(pareto.get_pareto_optimal(pathways))
        
        # Generate recommendations
        recommendations = []
        
        for rank, name in enumerate(mcda_result.rankings, 1):
            rec = Recommendation(
                pathway_name=name,
                rank=rank,
                score=mcda_result.scores.get(name, 0),
                is_pareto_optimal=name in pareto_optimal,
            )
            
            # Analyze strengths and weaknesses
            metrics = pathways.get(name, {})
            rec.strengths, rec.weaknesses = self._analyze_metrics(metrics)
            
            # Generate explanation
            rec.explanation = self._generate_explanation(rec, metrics)
            
            recommendations.append(rec)
        
        return recommendations
    
    def _analyze_metrics(
        self,
        metrics: Dict[str, float]
    ) -> tuple:
        """Identify strengths and weaknesses."""
        strengths = []
        weaknesses = []
        
        # Simple thresholds (would be better with benchmark data)
        if metrics.get("npv", 0) > 50:
            strengths.append("High NPV")
        elif metrics.get("npv", 0) < 0:
            weaknesses.append("Negative NPV")
        
        if metrics.get("gwp", float('inf')) < 100:
            strengths.append("Low carbon footprint")
        elif metrics.get("gwp", 0) > 500:
            weaknesses.append("High carbon footprint")
        
        if metrics.get("trl", 0) >= 8:
            strengths.append("Mature technology")
        elif metrics.get("trl", 0) <= 5:
            weaknesses.append("Low technology readiness")
        
        if metrics.get("overall_risk", 100) < 30:
            strengths.append("Low risk profile")
        elif metrics.get("overall_risk", 0) > 60:
            weaknesses.append("High risk profile")
        
        if metrics.get("irr", 0) > 0.15:
            strengths.append("Strong returns")
        
        if metrics.get("payback", float('inf')) < 5:
            strengths.append("Quick payback")
        elif metrics.get("payback", 0) > 10:
            weaknesses.append("Long payback period")
        
        return strengths, weaknesses
    
    def _generate_explanation(
        self,
        rec: Recommendation,
        metrics: Dict[str, float]
    ) -> str:
        """Generate human-readable explanation."""
        parts = []
        
        if rec.rank == 1:
            parts.append(f"{rec.pathway_name} ranks #1 with a score of {rec.score:.3f}.")
        else:
            parts.append(f"{rec.pathway_name} ranks #{rec.rank} with a score of {rec.score:.3f}.")
        
        if rec.is_pareto_optimal:
            parts.append("This pathway is Pareto-optimal (not dominated by any other option).")
        
        if rec.strengths:
            parts.append(f"Strengths: {', '.join(rec.strengths)}.")
        
        if rec.weaknesses:
            parts.append(f"Weaknesses: {', '.join(rec.weaknesses)}.")
        
        return " ".join(parts)
    
    def generate_summary(
        self,
        recommendations: List[Recommendation]
    ) -> Dict:
        """Generate summary of ranking results."""
        if not recommendations:
            return {"status": "No pathways to rank"}
        
        return {
            "best_pathway": recommendations[0].pathway_name,
            "best_score": recommendations[0].score,
            "pareto_optimal_count": sum(1 for r in recommendations if r.is_pareto_optimal),
            "pareto_optimal_pathways": [r.pathway_name for r in recommendations if r.is_pareto_optimal],
            "rankings": [r.to_dict() for r in recommendations],
        }
