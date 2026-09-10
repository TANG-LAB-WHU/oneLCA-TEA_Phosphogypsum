"""
Tests for Streamlit Dashboard integration and imports (v0.7.0).
"""

import pytest


def test_dashboard_import():
    from pgloop.visualization.dashboard import run_dashboard, main
    assert callable(run_dashboard)
    assert callable(main)


def test_dashboard_pathway_registry_alignment():
    from pgloop.pathways import PATHWAYS, list_pathways, get_pathway
    codes = list_pathways()
    assert len(codes) >= 7
    for code in codes:
        pw = get_pathway(code)
        assert pw.code == code
        assert isinstance(pw.name, str)


def test_dashboard_integrated_assessment_call():
    from pgloop.assessment import IntegratedAssessmentEngine
    engine = IntegratedAssessmentEngine(country="China")
    res = engine.assess("PG-CementProd", functional_unit_kg=1000.0)
    assert "lca" in res
    assert "tea" in res
    assert "risk" in res
    assert "impacts" in res["lca"]
    assert "clcc" in res["tea"]
    assert "overall_score" in res["risk"]
