"""Tests for joint uncertainty propagation."""

from pgloop.lca.lca_engine import LCAEngine
from pgloop.pathways import get_pathway
from pgloop.tea.tea_engine import TEAEngine
from pgloop.uncertainty.propagation import JointUncertaintyPropagator


def test_joint_propagation_summary_keys():
    lca = LCAEngine()
    tea = TEAEngine(country="China")
    pathway = get_pathway("PG-CementProd", country="China")

    propagator = JointUncertaintyPropagator(lca, tea, n_iterations=20, seed=1)
    result = propagator.propagate(pathway)

    assert result.n_samples == 20
    assert "gwp" in result.summary
    assert "clcc" in result.summary
    assert "slcc" in result.summary
    assert "lcop" in result.summary

