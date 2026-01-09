"""
Macro Risk - Policy Risk Assessment

Government policies: subsidies, carbon pricing, environmental regulations.
"""

from dataclasses import dataclass
from typing import Dict, List
from pgloop.risk.aggregator import RiskScore


@dataclass
class PolicyRisk:
    """
    Policy and regulatory risk assessment.
    """
    
    def assess(
        self,
        subsidy_dependency: float = 0.2,
        carbon_price_exposure: float = 0.0,
        environmental_permit_complexity: str = "moderate",
        policy_change_likelihood: str = "low",
        international_trade_exposure: float = 0.3
    ) -> RiskScore:
        """
        Assess policy risk.
        
        Args:
            subsidy_dependency: Fraction of revenue from subsidies
            carbon_price_exposure: Sensitivity to carbon pricing
            environmental_permit_complexity: Permit process (simple/moderate/complex)
            policy_change_likelihood: Risk of adverse policy change (low/medium/high)
            international_trade_exposure: Export/import dependency
            
        Returns:
            RiskScore for policy risk
        """
        factors = {}
        
        # Subsidy risk (high dependency = high risk if removed)
        subsidy_risk = min(80, subsidy_dependency * 200)
        factors["subsidy_risk"] = subsidy_risk
        
        # Carbon price risk (can be positive or negative)
        # For waste valorization, carbon price often helps
        if carbon_price_exposure < 0:  # Net carbon negative
            carbon_risk = max(0, 30 + carbon_price_exposure * 50)
        else:
            carbon_risk = min(70, carbon_price_exposure * 100)
        factors["carbon_risk"] = carbon_risk
        
        # Permit risk
        permit_map = {"simple": 15, "moderate": 35, "complex": 60}
        permit_risk = permit_map.get(environmental_permit_complexity, 35)
        factors["permit_risk"] = permit_risk
        
        # Policy change risk
        change_map = {"low": 15, "medium": 40, "high": 70}
        change_risk = change_map.get(policy_change_likelihood, 40)
        factors["policy_change_risk"] = change_risk
        
        # Trade policy risk
        if international_trade_exposure < 0.2:
            trade_risk = 10
        elif international_trade_exposure < 0.5:
            trade_risk = 30
        else:
            trade_risk = 50
        factors["trade_risk"] = trade_risk
        
        # Weighted combination
        score = (
            0.30 * subsidy_risk +
            0.20 * carbon_risk +
            0.20 * permit_risk +
            0.20 * change_risk +
            0.10 * trade_risk
        )
        
        mitigations = []
        if subsidy_risk > 40:
            mitigations.append("Reduce subsidy dependency")
        if permit_risk > 40:
            mitigations.append("Early stakeholder engagement")
        if change_risk > 40:
            mitigations.append("Policy advocacy / diversification")
        
        return RiskScore.from_score(
            category="policy",
            subcategory="regulatory",
            score=score,
            factors=factors,
            description=f"Subsidy_dep={subsidy_dependency:.0%}",
            mitigation=mitigations,
        )
