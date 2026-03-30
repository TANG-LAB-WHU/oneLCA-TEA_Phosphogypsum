"""Tests for Bayesian update closed-loop utilities."""

from pgloop.lca.lca_engine import LCAEngine
from pgloop.pathways import get_pathway
from pgloop.tea.tea_engine import TEAEngine
from pgloop.uncertainty.bayesian_update import BayesianUpdater


def test_bayesian_update_returns_posterior():
    updater = BayesianUpdater(observation_noise=0.2)
    priors = {"carbon_price_usd_t": {"type": "triangular", "min": 80, "mode": 100, "max": 140}}
    out = updater.update_priors(priors, observations={"carbon_price_usd_t": 120}, predictions={})
    assert "carbon_price_usd_t" in out.posterior


def test_bayesian_closed_loop_structure():
    lca = LCAEngine()
    tea = TEAEngine(country="China")
    pathway = get_pathway("PG-CementProd", country="China")
    updater = BayesianUpdater(observation_noise=0.2)
    priors = {"carbon_price_usd_t": {"type": "triangular", "min": 80, "mode": 100, "max": 140}}
    result = updater.run_closed_loop(
        pathway=pathway,
        lca_engine=lca,
        tea_engine=tea,
        priors=priors,
        observations={"gwp": 120.0, "clcc": 45.0},
        n_iterations=15,
    )
    assert "first_pass" in result
    assert "second_pass" in result

