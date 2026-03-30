"""
PG-LCA-TEA Main Entry Point

Demonstrates the integrated assessment workflow:
1. Pathway initialization
2. LCA (Environmental Impact)
3. TEA (Economic Assessment)
4. Risk Assessment (Macro & Micro)
5. Decision Support (Ranking & Recommendation)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

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

    # 2. Select Pathways
    pathway_codes = ["PG-Stack", "PG-CementProd", "PG-REEextract"]
    pathways = [get_pathway(code) for code in pathway_codes]

    print(f"\nAnalyzing {len(pathways)} pathways...")

    # Storage for decision metrics
    decision_data = {}

    for pathway in pathways:
        print(f"\n--- Processing: {pathway.name} [{pathway.code}] ---")

        # A. LCA Calculation
        lca_result = lca_engine.calculate(pathway, functional_unit_value=1.0)
        gwp = lca_result.impacts.get("climate_change", 0)
        print(f"  LCA: GWP = {gwp:.2f} kg CO2-eq/t")

        # B. TEA Calculation
        tea_result = tea_engine.calculate(pathway, functional_unit_value=1.0)
        npv_result = tea_engine.calculate_npv(pathway)
        npv = npv_result.get("npv", 0)
        payback = npv_result.get("payback_years", 20)

        print(f"  TEA: CLCC = {format_currency(tea_result.clcc)}/t")
        print(f"  TEA: NPV  = {format_currency(npv)}")

        # C. Risk Assessment (Sample Scores)
        # Note: In a real run, these would come from the risk.macro and risk.micro modules
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

        # D. Collect metrics for ranking
        decision_data[pathway.name] = {
            "gwp": gwp,
            "resource_depletion": lca_result.impacts.get("resource_depletion", 0),
            "human_toxicity": lca_result.impacts.get("human_toxicity", 0),
            "npv": npv / 1000000,  # normalized in Millions for scoring
            "irr": 0.15,  # Sample IRR
            "payback": payback,
            "trl": pathway.trl,
            "scalability": 0.8,  # Assumed
            "overall_risk": aggregated_risk.overall_score,
        }

    # 3. Decision Support (Ranking)
    print("\n" + "=" * 60)
    print("   MULTI-CRITERIA DECISION ANALYSIS (MCDA) RESULTS   ")
    print("=" * 60)

    recommendations = pathway_ranker.rank(decision_data)

    for rec in recommendations:
        status = " [OPTIMAL]" if rec.is_pareto_optimal else ""
        print(f"\nRank {rec.rank}: {rec.pathway_name}{status}")
        print(f"  Score: {rec.score:.3f}")
        print(f"  Explanation: {rec.explanation}")

    # 4. Scenario Analysis (Sensitivity to REE price)
    print("\n" + "=" * 60)
    print("   SCENARIO ANALYSIS: MARKET ROBUSTNESS   ")
    print("=" * 60)

    analyzer = ScenarioAnalyzer(lca_engine, tea_engine)

    # We'll use the Cement pathway for scenario testing
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
