"""
Macro Risk - Political Risk Assessment

Political stability, regulatory environment, governance.
"""

from dataclasses import dataclass

from pgloop.risk.aggregator import RiskScore

# Country political risk indices (0-100, higher = more stable)
POLITICAL_STABILITY_INDEX = {
    "USA": 75,
    "Canada": 85,
    "Germany": 85,
    "France": 75,
    "UK": 80,
    "Japan": 85,
    "Australia": 85,
    "China": 60,
    "India": 55,
    "Brazil": 50,
    "Russia": 35,
    "Morocco": 55,
    "Tunisia": 45,
    "Saudi Arabia": 50,
    "South Africa": 45,
}


@dataclass
class PoliticalRisk:
    """
    Macro-level political risk assessment.
    """

    def assess(
        self,
        country: str,
        regulatory_stability: str = "stable",
        expropriation_risk: str = "low",
        corruption_level: str = "moderate",
    ) -> RiskScore:
        """
        Assess political risk for a country.

        Args:
            country: Country name
            regulatory_stability: Regulatory environment (volatile/changing/stable)
            expropriation_risk: Risk of asset seizure (high/medium/low)
            corruption_level: Corruption perception (high/moderate/low)

        Returns:
            RiskScore for political risk
        """
        factors = {}

        # Country stability (invert: higher stability = lower risk)
        stability_score = POLITICAL_STABILITY_INDEX.get(country, 50)
        country_risk = 100 - stability_score
        factors["country_stability"] = country_risk

        # Regulatory stability
        reg_map = {"volatile": 70, "changing": 40, "stable": 15}
        reg_risk = reg_map.get(regulatory_stability, 40)
        factors["regulatory_risk"] = reg_risk

        # Expropriation risk
        exprop_map = {"high": 80, "medium": 40, "low": 10}
        exprop_risk = exprop_map.get(expropriation_risk, 40)
        factors["expropriation_risk"] = exprop_risk

        # Corruption risk
        corrupt_map = {"high": 60, "moderate": 30, "low": 10}
        corrupt_risk = corrupt_map.get(corruption_level, 30)
        factors["corruption_risk"] = corrupt_risk

        # Weighted combination
        score = 0.35 * country_risk + 0.25 * reg_risk + 0.25 * exprop_risk + 0.15 * corrupt_risk

        mitigations = []
        if country_risk > 40:
            mitigations.append("Political risk insurance")
        if exprop_risk > 30:
            mitigations.append("International investment treaties")
        if corrupt_risk > 40:
            mitigations.append("Robust compliance program")

        return RiskScore.from_score(
            category="political",
            subcategory="country_risk",
            score=score,
            factors=factors,
            description=f"Country: {country}",
            mitigation=mitigations,
        )
