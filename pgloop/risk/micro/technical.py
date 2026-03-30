"""
Micro Risk - Technical Risk Assessment

Technology readiness, reliability, and scalability risks.
"""

from dataclasses import dataclass

from pgloop.risk.aggregator import RiskScore


@dataclass
class TechnicalRisk:
    """
    Technical risk assessment for a treatment pathway.

    Evaluates technology maturity, reliability, and scale-up risk.
    """

    def assess(
        self,
        trl: int,
        scale_factor: float = 1.0,
        complexity: str = "medium",
        novel_technology: bool = False,
    ) -> RiskScore:
        """
        Assess technical risk.

        Args:
            trl: Technology Readiness Level (1-9)
            scale_factor: Ratio of target scale to demonstration scale
            complexity: Process complexity (low/medium/high)
            novel_technology: Whether technology is novel/unproven

        Returns:
            RiskScore for technical risk
        """
        factors = {}

        # TRL risk (lower TRL = higher risk)
        trl_risk = max(0, (9 - trl) * 10)  # 0-80
        factors["trl_risk"] = trl_risk

        # Scale-up risk
        if scale_factor < 2:
            scale_risk = 10
        elif scale_factor < 10:
            scale_risk = 30
        elif scale_factor < 100:
            scale_risk = 50
        else:
            scale_risk = 70
        factors["scale_risk"] = scale_risk

        # Complexity risk
        complexity_map = {"low": 10, "medium": 30, "high": 50}
        complexity_risk = complexity_map.get(complexity, 30)
        factors["complexity_risk"] = complexity_risk

        # Novelty risk
        novelty_risk = 30 if novel_technology else 0
        factors["novelty_risk"] = novelty_risk

        # Weighted combination
        score = 0.35 * trl_risk + 0.30 * scale_risk + 0.20 * complexity_risk + 0.15 * novelty_risk

        mitigations = []
        if trl_risk > 30:
            mitigations.append("Conduct pilot-scale demonstration")
        if scale_risk > 30:
            mitigations.append("Staged scale-up approach")
        if complexity_risk > 30:
            mitigations.append("Simplify process design")

        return RiskScore.from_score(
            category="technical",
            subcategory="technology_readiness",
            score=score,
            factors=factors,
            description=f"TRL={trl}, Scale={scale_factor}x",
            mitigation=mitigations,
        )
