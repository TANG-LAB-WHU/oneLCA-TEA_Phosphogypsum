"""
Macro Risk - Economic Risk Assessment

Macroeconomic factors: monetary policy, credit/loan conditions, 
exchange rates, inflation, interest rates, and GDP.
"""

from dataclasses import dataclass
from typing import Dict
from pgloop.risk.aggregator import RiskScore


# Country credit ratings (simplified S&P-style)
COUNTRY_CREDIT_RATINGS = {
    "USA": "AAA", "Germany": "AAA", "Canada": "AAA", "Australia": "AAA",
    "UK": "AA", "France": "AA", "Japan": "A", "China": "A",
    "India": "BBB", "Brazil": "BB", "Russia": "B", "Morocco": "BB",
}

RATING_RISK_MAP = {
    "AAA": 5, "AA+": 10, "AA": 15, "AA-": 20,
    "A+": 25, "A": 30, "A-": 35,
    "BBB+": 40, "BBB": 45, "BBB-": 50,
    "BB+": 55, "BB": 60, "BB-": 65,
    "B+": 70, "B": 75, "B-": 80,
    "CCC": 90, "CC": 95, "D": 100,
}


@dataclass
class EconomicRisk:
    """
    Macroeconomic risk assessment including monetary policy,
    credit conditions, and exchange rate factors.
    """
    
    def assess(
        self,
        country: str = "USA",
        # Monetary Policy
        central_bank_independence: str = "high",  # low/medium/high
        monetary_policy_stance: str = "neutral",  # tight/neutral/loose
        money_supply_growth: float = 0.05,
        # Credit/Loan Factors
        credit_availability: str = "adequate",  # restricted/adequate/abundant
        lending_rate: float = 0.06,
        lending_rate_spread: float = 0.02,  # Over risk-free rate
        credit_rating: str = None,  # Will lookup if None
        # Exchange Rate
        currency_volatility: float = 0.10,
        currency_regime: str = "floating",  # fixed/managed/floating
        fx_reserves_months: float = 6.0,  # Import coverage
        # General Macro
        inflation_rate: float = 0.03,
        gdp_growth: float = 0.02,
    ) -> RiskScore:
        """
        Assess macroeconomic risk with enhanced factors.
        
        Args:
            country: Country name
            central_bank_independence: CB autonomy level
            monetary_policy_stance: Current monetary policy
            money_supply_growth: M2 growth rate
            credit_availability: Credit market conditions
            lending_rate: Commercial lending rate
            lending_rate_spread: Spread over base rate
            credit_rating: Sovereign credit rating
            currency_volatility: FX volatility (annualized)
            currency_regime: Exchange rate regime
            fx_reserves_months: FX reserves in months of imports
            inflation_rate: CPI inflation
            gdp_growth: Real GDP growth
            
        Returns:
            RiskScore for economic risk
        """
        factors = {}
        
        # === Monetary Policy Risk ===
        # Central bank independence
        cb_map = {"low": 60, "medium": 35, "high": 15}
        cb_risk = cb_map.get(central_bank_independence, 35)
        factors["cb_independence_risk"] = cb_risk
        
        # Monetary policy stance
        if monetary_policy_stance == "tight":
            mp_risk = 40  # Restrictive environment
        elif monetary_policy_stance == "neutral":
            mp_risk = 20
        else:  # loose
            mp_risk = 35  # Potential inflation/bubble
        factors["monetary_policy_risk"] = mp_risk
        
        # Money supply growth (extreme values = risk)
        if money_supply_growth < 0:
            ms_risk = 60  # Deflationary
        elif money_supply_growth < 0.03:
            ms_risk = 30
        elif money_supply_growth < 0.10:
            ms_risk = 20
        elif money_supply_growth < 0.20:
            ms_risk = 40
        else:
            ms_risk = 70  # Hyperinflation risk
        factors["money_supply_risk"] = ms_risk
        
        # === Credit/Loan Risk ===
        # Credit availability
        credit_map = {"restricted": 65, "adequate": 25, "abundant": 15}
        credit_risk = credit_map.get(credit_availability, 25)
        factors["credit_availability_risk"] = credit_risk
        
        # Lending rate level
        if lending_rate < 0.04:
            lr_risk = 15
        elif lending_rate < 0.08:
            lr_risk = 25
        elif lending_rate < 0.15:
            lr_risk = 45
        else:
            lr_risk = 70
        factors["lending_rate_risk"] = lr_risk
        
        # Credit spread (higher = higher risk perception)
        spread_risk = min(70, lending_rate_spread * 1500)
        factors["credit_spread_risk"] = spread_risk
        
        # Sovereign credit rating
        if credit_rating is None:
            credit_rating = COUNTRY_CREDIT_RATINGS.get(country, "BBB")
        rating_risk = RATING_RISK_MAP.get(credit_rating, 50)
        factors["sovereign_rating_risk"] = rating_risk
        
        # === Exchange Rate Risk ===
        # Currency volatility
        fx_vol_risk = min(80, currency_volatility * 400)
        factors["fx_volatility_risk"] = fx_vol_risk
        
        # Currency regime
        regime_map = {"fixed": 40, "managed": 25, "floating": 30}
        regime_risk = regime_map.get(currency_regime, 30)
        factors["fx_regime_risk"] = regime_risk
        
        # FX reserves adequacy
        if fx_reserves_months < 3:
            reserve_risk = 70
        elif fx_reserves_months < 6:
            reserve_risk = 40
        elif fx_reserves_months < 12:
            reserve_risk = 20
        else:
            reserve_risk = 10
        factors["fx_reserve_risk"] = reserve_risk
        
        # === General Macro Risk ===
        # Inflation risk
        if inflation_rate < 0.02:
            inflation_risk = 15
        elif inflation_rate < 0.05:
            inflation_risk = 25
        elif inflation_rate < 0.10:
            inflation_risk = 50
        else:
            inflation_risk = 75
        factors["inflation_risk"] = inflation_risk
        
        # GDP growth risk
        if gdp_growth > 0.03:
            gdp_risk = 15
        elif gdp_growth > 0.01:
            gdp_risk = 25
        elif gdp_growth > 0:
            gdp_risk = 40
        else:
            gdp_risk = 70
        factors["gdp_risk"] = gdp_risk
        
        # === Weighted Aggregation ===
        # Group weights
        monetary_score = 0.25 * cb_risk + 0.35 * mp_risk + 0.40 * ms_risk
        credit_score = 0.30 * credit_risk + 0.25 * lr_risk + 0.20 * spread_risk + 0.25 * rating_risk
        fx_score = 0.50 * fx_vol_risk + 0.20 * regime_risk + 0.30 * reserve_risk
        macro_score = 0.50 * inflation_risk + 0.50 * gdp_risk
        
        # Overall score
        score = (
            0.20 * monetary_score +
            0.30 * credit_score +
            0.30 * fx_score +
            0.20 * macro_score
        )
        
        # Mitigations
        mitigations = []
        if fx_vol_risk > 40:
            mitigations.append("Currency hedging (forwards/options)")
        if credit_risk > 40:
            mitigations.append("Diversify financing sources")
        if lr_risk > 40:
            mitigations.append("Lock in fixed-rate financing")
        if rating_risk > 50:
            mitigations.append("Obtain credit enhancement / guarantees")
        if cb_risk > 40:
            mitigations.append("Monitor monetary policy developments")
        
        return RiskScore.from_score(
            category="economic",
            subcategory="macroeconomic",
            score=score,
            factors=factors,
            description=f"{country}: Inflation={inflation_rate:.1%}, FX_vol={currency_volatility:.0%}, Rate={credit_rating}",
            mitigation=mitigations,
        )

