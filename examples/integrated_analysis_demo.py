"""
Integrated Multi-Pathway Analysis Demo for PhosphogypsumBot.

Demonstrates the full integrated assessment workflow across multiple pathways:
1. Pathway initialization & default parameter loading.
2. Life Cycle Assessment (LCA) environmental calculations.
3. Techno-Economic Analysis (TEA) financial calculations & NPV.
4. Risk Assessment aggregation (process and market/policy risks).
5. Decision Support using MCDA (TOPSIS/AHP) ranking.
6. Market Robustness Scenario Analysis.
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pgloop import LCAEngine, PathwayRanker, RiskAggregator, TEAEngine, get_pathway
from pgloop.decision.scenario import MARKET_SCENARIOS, ScenarioAnalyzer
from pgloop.risk.aggregator import RiskScore
from pgloop.utils.currency import format_currency


def run_integrated_analysis():
    print("=" * 60)
    print("   PHOSPHOGYPSUM INTEGRATED ASSESSMENT FRAMEWORK (PGLOOP)   ")
    print("=" * 60)

    # 1. Initialize Engines
    lca_engine = LCAEngine()
    tea_engine = TEAEngine(country="China")
    risk_aggregator = RiskAggregator()
    pathway_ranker = PathwayRanker()

    # 2. Select Pathways for evaluation
    pathway_codes = ["PG-Stack", "PG-CementProd", "PG-REEextract"]
    pathways = [get_pathway(code) for code in pathway_codes]

    print(f"\nAnalyzing {len(pathways)} pathways...")

    # Storage for decision metrics
    decision_data = {}

    for pathway in pathways:
        print(f"\n--- Processing: {pathway.name} [{pathway.code}] ---")

        # A. LCA Calculation
        lca_result = lca_engine.calculate(pathway, functional_unit_value=1.0)
        gwp = lca_result.impacts.get("climate_change", 0.0)
        print(f"  LCA: GWP = {gwp:.2f} kg CO2-eq/t")

        # B. TEA Calculation
        tea_result = tea_engine.calculate(pathway, functional_unit_value=1.0)
        npv_result = tea_engine.calculate_npv(pathway)
        npv = npv_result.get("npv", 0.0)
        payback = npv_result.get("payback_years", 20.0)

        print(f"  TEA: CLCC = {format_currency(tea_result.clcc)}/t")
        print(f"  TEA: NPV  = {format_currency(npv)}")

        # C. Risk Assessment aggregation (Sample Scores)
        sample_risks = [
            RiskScore.from_score(
                "technical", "tech_maturity", 100 - (pathway.trl * 10), description="Based on TRL"
            ),
            RiskScore.from_score("economic", "price_volatility", 35, description="Market risk"),
            RiskScore.from_score(
                "policy", "regulatory_stringency", 45, description="Environmental law"
            ),
        ]
        aggregated_risk = risk_aggregator.aggregate(sample_risks)
        risk_level = aggregated_risk.overall_level.name
        print(f"  Risk: Score = {aggregated_risk.overall_score:.2f} [{risk_level}]")

        # D. Collect metrics for MCDA ranking
        decision_data[pathway.name] = {
            "gwp": gwp,
            "resource_depletion": lca_result.impacts.get("resource_depletion", 0.0),
            "human_toxicity": lca_result.impacts.get("human_toxicity", 0.0),
            "npv": npv / 1000000.0,  # normalized in Millions for scoring
            "irr": 0.15,  # Sample IRR
            "payback": payback,
            "trl": pathway.trl,
            "scalability": 0.8,  # Assumed
            "overall_risk": aggregated_risk.overall_score,
        }

    # 3. Decision Support (TOPSIS/AHP Ranking)
    print("\n" + "=" * 60)
    print("   MULTI-CRITERIA DECISION ANALYSIS (MCDA) RESULTS   ")
    print("=" * 60)

    recommendations = pathway_ranker.rank(decision_data)

    for rec in recommendations:
        status = " [OPTIMAL]" if rec.is_pareto_optimal else ""
        print(f"\nRank {rec.rank}: {rec.pathway_name}{status}")
        print(f"  Score: {rec.score:.3f}")
        print(f"  Explanation: {rec.explanation}")

    # 4. Scenario Analysis (Sensitivity to market/price conditions)
    print("\n" + "=" * 60)
    print("   SCENARIO ANALYSIS: MARKET ROBUSTNESS   ")
    print("=" * 60)

    analyzer = ScenarioAnalyzer(lca_engine, tea_engine)

    # Use the Cement pathway for scenario testing
    cement_pathway = get_pathway("PG-CementProd")

    robustness = analyzer.quick_robustness_check(
        pathway=cement_pathway,
        scenarios=list(MARKET_SCENARIOS.values())[:3],  # Baseline, Optimistic, Pessimistic
        metric="clcc",
    )

    for scenario_name, value in robustness.items():
        if scenario_name != "robustness_stats":
            print(f"  {scenario_name:12}: {format_currency(value)}/t")

    print("\nFinished Integrated Analysis.")


def main():
    run_integrated_analysis()


if __name__ == "__main__":
    main()
