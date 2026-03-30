"""
Micro Risk - Project Financial Risk Assessment

Project-level financial risks: funding, cash flow, cost overruns.
"""

from dataclasses import dataclass

from pgloop.risk.aggregator import RiskScore


@dataclass
class ProjectFinancialRisk:
    """
    Project-level financial risk assessment.
    """

    def assess(
        self,
        debt_ratio: float = 0.6,
        irr: float = 0.15,
        payback_years: float = 7.0,
        capex_contingency: float = 0.15,
        revenue_contract_coverage: float = 0.5,
    ) -> RiskScore:
        """
        Assess project financial risk.

        Args:
            debt_ratio: Debt-to-equity ratio
            irr: Internal rate of return
            payback_years: Payback period in years
            capex_contingency: CAPEX contingency allowance
            revenue_contract_coverage: Fraction of revenue under contract

        Returns:
            RiskScore for financial risk
        """
        factors = {}

        # Leverage risk
        if debt_ratio < 0.4:
            leverage_risk = 10
        elif debt_ratio < 0.7:
            leverage_risk = 25
        elif debt_ratio < 0.9:
            leverage_risk = 50
        else:
            leverage_risk = 75
        factors["leverage_risk"] = leverage_risk

        # Return risk
        if irr > 0.20:
            return_risk = 10
        elif irr > 0.15:
            return_risk = 25
        elif irr > 0.10:
            return_risk = 40
        elif irr > 0.05:
            return_risk = 60
        else:
            return_risk = 80
        factors["return_risk"] = return_risk

        # Payback risk
        if payback_years < 5:
            payback_risk = 15
        elif payback_years < 8:
            payback_risk = 30
        elif payback_years < 12:
            payback_risk = 50
        else:
            payback_risk = 70
        factors["payback_risk"] = payback_risk

        # Cost overrun risk (lower contingency = higher risk)
        overrun_risk = max(10, 60 - capex_contingency * 200)
        factors["overrun_risk"] = overrun_risk

        # Revenue certainty
        revenue_risk = max(10, 70 - revenue_contract_coverage * 70)
        factors["revenue_risk"] = revenue_risk

        # Weighted combination
        score = (
            0.20 * leverage_risk
            + 0.25 * return_risk
            + 0.20 * payback_risk
            + 0.15 * overrun_risk
            + 0.20 * revenue_risk
        )

        mitigations = []
        if leverage_risk > 40:
            mitigations.append("Increase equity financing")
        if return_risk > 40:
            mitigations.append("Value engineering to improve returns")
        if revenue_risk > 40:
            mitigations.append("Secure offtake agreements")

        return RiskScore.from_score(
            category="financial",
            subcategory="project_finance",
            score=score,
            factors=factors,
            description=f"IRR={irr:.1%}, Payback={payback_years}yr",
            mitigation=mitigations,
        )
