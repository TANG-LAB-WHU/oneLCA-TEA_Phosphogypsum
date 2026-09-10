"""
Integrated Assessment Module

Provides the high-level facade IntegratedAssessmentEngine for end-to-end
multi-dimensional sustainability assessment, combining LCA, TEA, VPM kinetics,
Risk aggregation, and MCDA pathway ranking.
"""

from typing import Any, Dict, List, Optional, Union

from pgloop.decision.recommender import PathwayRanker, Recommendation
from pgloop.lca.lca_engine import LCAEngine
from pgloop.pathways import BasePathway, get_pathway, list_pathways
from pgloop.risk.aggregator import RiskAggregator, RiskScore
from pgloop.tea.tea_engine import TEAEngine


class IntegratedAssessmentEngine:
    """
    Unified evaluation engine combining Life Cycle Assessment (LCA),
    Techno-Economic Analysis (TEA), physics-based Valorization Pathway Modules (VPM),
    risk aggregation, and multi-criteria decision ranking (MCDA).
    """

    def __init__(
        self,
        country: str = "China",
        mcda_method: str = "TOPSIS",
    ):
        self.country = country
        self.lca_engine = LCAEngine()
        self.tea_engine = TEAEngine(country=country)
        self.risk_aggregator = RiskAggregator()
        self.ranker = PathwayRanker(method=mcda_method)

    def assess(
        self,
        pathway_or_code: Union[str, BasePathway],
        functional_unit_kg: float = 1000.0,
        operating_conditions: Optional[Dict[str, Any]] = None,
        include_external: bool = True,
        include_uncertainty: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute full multi-dimensional assessment of a single pathway.

        Args:
            pathway_or_code: Pathway instance or valid pathway code string.
            functional_unit_kg: Functional unit mass in kg (default: 1000.0 kg / 1 tonne).
            operating_conditions: Optional dynamic operating conditions for VPM simulation.
            include_external: Whether to calculate external environmental costs in TEA.
            include_uncertainty: Whether to calculate uncertainty metrics.

        Returns:
            Dictionary containing pathway metadata, LCA results, TEA results, VPM simulation, and Risk.
        """
        if isinstance(pathway_or_code, str):
            pathway = get_pathway(pathway_or_code)
        else:
            pathway = pathway_or_code

        fu_val = functional_unit_kg / 1000.0  # normalize to pathway functional unit (tonnes)

        # 1. LCA Calculation
        lca_res = self.lca_engine.calculate(pathway, functional_unit_value=fu_val)

        # 2. TEA Calculation
        tea_res = self.tea_engine.calculate(
            pathway,
            functional_unit_value=fu_val,
            include_external=include_external,
            include_uncertainty=include_uncertainty,
        )
        npv_res = self.tea_engine.calculate_npv(pathway)

        # 3. Dynamic VPM Kinetics Simulation
        vpm_res = pathway.evaluate_operating_conditions(operating_conditions)

        # 4. Risk Assessment
        trl = getattr(pathway, "trl", 7)
        tech_risk = max(0.0, min(100.0, 100.0 - (trl * 10.0)))
        risk_scores = [
            RiskScore.from_score("technical", "tech_maturity", tech_risk, description=f"TRL {trl}"),
            RiskScore.from_score("economic", "price_volatility", 35.0, description="Market risk"),
            RiskScore.from_score("policy", "regulatory_stringency", 40.0, description="Environmental compliance"),
        ]
        agg_risk = self.risk_aggregator.aggregate(risk_scores)

        return {
            "pathway_code": pathway.code,
            "pathway_name": pathway.name,
            "trl": trl,
            "functional_unit_kg": functional_unit_kg,
            "lca": {
                "impacts": lca_res.impacts,
                "normalized_impacts": lca_res.normalized_impacts,
                "uncertainty": lca_res.uncertainty,
            },
            "tea": {
                "clcc": tea_res.clcc,
                "slcc": tea_res.slcc,
                "capex_total": tea_res.capex_total,
                "capex_annualized": tea_res.capex_annualized,
                "opex_total": tea_res.opex_total,
                "revenue": tea_res.revenue,
                "external_cost": tea_res.external_cost,
                "npv": npv_res.get("npv", 0.0),
                "irr": npv_res.get("irr", 0.0),
                "payback_years": npv_res.get("payback_years", 20.0),
            },
            "vpm": vpm_res,
            "risk": {
                "overall_score": agg_risk.overall_score,
                "overall_level": agg_risk.overall_level.name,
                "risk_adjusted_discount_rate": agg_risk.risk_adjusted_discount_rate,
            },
        }

    def rank_all_pathways(
        self,
        pathway_codes: Optional[List[str]] = None,
        functional_unit_kg: float = 1000.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Recommendation]:
        """
        Rank pathways using Multi-Criteria Decision Analysis (MCDA) across 5D criteria.

        Args:
            pathway_codes: List of pathway codes to evaluate. Defaults to all registered pathways.
            functional_unit_kg: Functional unit mass in kg.
            weights: Optional dictionary of category weights ("environmental", "economic", "risk", "social").

        Returns:
            List of Recommendation objects ordered from best to worst.
        """
        codes = pathway_codes or list_pathways()
        decision_data = {}

        if weights:
            self.ranker = PathwayRanker(
                method=self.ranker.method,
                lca_weight=weights.get("environmental", 0.25),
                tea_weight=weights.get("economic", 0.35),
                risk_weight=weights.get("risk", 0.15),
                social_weight=weights.get("social", 0.10),
            )

        for code in codes:
            try:
                assessment = self.assess(code, functional_unit_kg=functional_unit_kg)
                name = assessment["pathway_name"]
                lca_impacts = assessment["lca"]["impacts"]
                tea_metrics = assessment["tea"]
                risk_metrics = assessment["risk"]
                trl = assessment["trl"]

                # Scalability computed from TRL (scale 0-1)
                scalability = min(1.0, max(0.1, trl / 9.0))

                decision_data[name] = {
                    "gwp": lca_impacts.get("climate_change", 100.0),
                    "resource_depletion": lca_impacts.get("resource_depletion", 0.0),
                    "human_toxicity": lca_impacts.get("human_toxicity", 0.0),
                    "npv": tea_metrics["npv"] / 1_000_000.0,  # In Millions USD
                    "irr": tea_metrics["irr"],
                    "payback": tea_metrics["payback_years"],
                    "trl": trl,
                    "scalability": scalability,
                    "overall_risk": risk_metrics["overall_score"],
                }
            except Exception:
                continue

        return self.ranker.rank(decision_data)
