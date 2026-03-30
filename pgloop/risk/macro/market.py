"""
Macro Risk - Market Risk Assessment

Product market dynamics: demand, price volatility, competition.
"""

from dataclasses import dataclass

from pgloop.risk.aggregator import RiskScore


@dataclass
class MarketRisk:
    """
    Market risk assessment for products.
    """

    def assess(
        self,
        price_volatility: float = 0.15,
        demand_trend: str = "stable",
        competition_intensity: str = "moderate",
        substitute_availability: str = "limited",
        market_concentration: float = 0.3,
    ) -> RiskScore:
        """
        Assess market risk for products.

        Args:
            price_volatility: Historical price volatility (CV)
            demand_trend: Demand trajectory (declining/stable/growing)
            competition_intensity: Competition level (low/moderate/high)
            substitute_availability: Substitute products (none/limited/many)
            market_concentration: HHI or top-4 concentration ratio

        Returns:
            RiskScore for market risk
        """
        factors = {}

        # Price volatility risk
        if price_volatility < 0.10:
            price_risk = 15
        elif price_volatility < 0.20:
            price_risk = 30
        elif price_volatility < 0.35:
            price_risk = 50
        else:
            price_risk = 70
        factors["price_volatility"] = price_risk

        # Demand risk
        demand_map = {"declining": 70, "stable": 25, "growing": 10}
        demand_risk = demand_map.get(demand_trend, 25)
        factors["demand_risk"] = demand_risk

        # Competition risk
        comp_map = {"low": 15, "moderate": 35, "high": 60}
        comp_risk = comp_map.get(competition_intensity, 35)
        factors["competition_risk"] = comp_risk

        # Substitute risk
        sub_map = {"none": 10, "limited": 30, "many": 55}
        sub_risk = sub_map.get(substitute_availability, 30)
        factors["substitute_risk"] = sub_risk

        # Market concentration (high concentration = buyer power risk)
        if market_concentration > 0.6:
            conc_risk = 50  # Oligopsony
        elif market_concentration > 0.3:
            conc_risk = 30
        else:
            conc_risk = 15
        factors["concentration_risk"] = conc_risk

        # Weighted combination
        score = (
            0.30 * price_risk
            + 0.25 * demand_risk
            + 0.20 * comp_risk
            + 0.15 * sub_risk
            + 0.10 * conc_risk
        )

        mitigations = []
        if price_risk > 40:
            mitigations.append("Price hedging / forward contracts")
        if demand_risk > 40:
            mitigations.append("Market diversification")
        if comp_risk > 40:
            mitigations.append("Cost leadership / differentiation strategy")

        return RiskScore.from_score(
            category="market",
            subcategory="product_market",
            score=score,
            factors=factors,
            description=f"Price_vol={price_volatility:.0%}, Demand={demand_trend}",
            mitigation=mitigations,
        )
