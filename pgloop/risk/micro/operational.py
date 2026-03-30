"""
Micro Risk - Operational Risk Assessment

Production capacity, quality consistency, and supply chain risks.
"""

from dataclasses import dataclass

from pgloop.risk.aggregator import RiskScore


@dataclass
class OperationalRisk:
    """
    Operational risk assessment for plant operations.
    """

    def assess(
        self,
        capacity_utilization: float = 0.8,
        feedstock_reliability: str = "high",
        product_quality_variance: float = 0.05,
        workforce_availability: str = "adequate",
    ) -> RiskScore:
        """
        Assess operational risk.

        Args:
            capacity_utilization: Target capacity utilization (0-1)
            feedstock_reliability: PG supply reliability (low/medium/high)
            product_quality_variance: CV of product quality
            workforce_availability: Labor market (scarce/adequate/abundant)

        Returns:
            RiskScore for operational risk
        """
        factors = {}

        # Capacity risk (too high or too low is risky)
        if capacity_utilization < 0.6:
            capacity_risk = 40  # Under-utilization
        elif capacity_utilization > 0.95:
            capacity_risk = 50  # Over-capacity stress
        else:
            capacity_risk = 20
        factors["capacity_risk"] = capacity_risk

        # Feedstock risk
        feedstock_map = {"low": 60, "medium": 30, "high": 10}
        feedstock_risk = feedstock_map.get(feedstock_reliability, 30)
        factors["feedstock_risk"] = feedstock_risk

        # Quality risk
        if product_quality_variance < 0.05:
            quality_risk = 10
        elif product_quality_variance < 0.10:
            quality_risk = 30
        elif product_quality_variance < 0.20:
            quality_risk = 50
        else:
            quality_risk = 70
        factors["quality_risk"] = quality_risk

        # Workforce risk
        workforce_map = {"scarce": 50, "adequate": 20, "abundant": 10}
        workforce_risk = workforce_map.get(workforce_availability, 20)
        factors["workforce_risk"] = workforce_risk

        # Weighted combination
        score = (
            0.25 * capacity_risk
            + 0.30 * feedstock_risk
            + 0.25 * quality_risk
            + 0.20 * workforce_risk
        )

        mitigations = []
        if feedstock_risk > 30:
            mitigations.append("Diversify feedstock sources")
        if quality_risk > 30:
            mitigations.append("Implement quality management system")
        if workforce_risk > 30:
            mitigations.append("Training and retention programs")

        return RiskScore.from_score(
            category="operational",
            subcategory="operations",
            score=score,
            factors=factors,
            mitigation=mitigations,
        )
