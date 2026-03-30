"""Tests for dynamic scenario trajectory support."""

from pgloop.decision.scenario import DynamicScenarioAnalyzer, Scenario
from pgloop.lca.lca_engine import LCAEngine
from pgloop.pathways import get_pathway
from pgloop.tea.tea_engine import TEAEngine


def test_scenario_trajectory_value_lookup():
    scenario = Scenario(
        name="test",
        trajectory={"carbon_price_usd_t": {2025: 100.0, 2030: 150.0}},
    )
    assert scenario.get_trajectory_value("carbon_price_usd_t", 2025) == 100.0
    # nearest-year fallback
    assert scenario.get_trajectory_value("carbon_price_usd_t", 2027) == 100.0


def test_dynamic_analyzer_runs_yearly():
    lca = LCAEngine()
    tea = TEAEngine(country="China")
    analyzer = DynamicScenarioAnalyzer(lca, tea)
    pathway = get_pathway("PG-CementProd", country="China")
    scenario = Scenario(
        name="baseline",
        trajectory={"carbon_price_usd_t": {2025: 100.0, 2026: 105.0}},
    )
    results = analyzer.run(pathway, scenario, start_year=2025, end_year=2026, n_samples=10)
    assert len(results) == 2
    assert results[0].year == 2025
    assert "clcc" in results[0].metrics
