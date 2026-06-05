from pgloop.decision import (
    BenefitCompensationModel,
    PathwayRanker,
    ReverseDesignOptimizer,
    create_default_criteria,
)


def test_social_criteria_in_defaults():
    criteria = create_default_criteria()
    # Check that social criteria are present
    job_c = criteria.get_by_name("job_creation")
    health_r = criteria.get_by_name("community_health_risk")

    assert job_c is not None
    assert job_c.category.value == "social"
    assert job_c.weight > 0

    assert health_r is not None
    assert health_r.category.value == "social"
    assert health_r.weight > 0


def test_pathway_ranker_supports_social_weights():
    # Construct ranker with explicit social weight
    ranker = PathwayRanker(
        lca_weight=0.20,
        tea_weight=0.30,
        risk_weight=0.20,
        social_weight=0.15,
    )
    assert ranker.category_weights["social"] == 0.15

    # Mock pathways with social metrics
    pathways = {
        "PG-CementProd": {
            "npv": 45.0,
            "gwp": 120.0,
            "trl": 9,
            "overall_risk": 20.0,
            "job_creation": 5.0,
            "community_health_risk": 15.0,
        },
        "PG-REEextract": {
            "npv": 75.0,
            "gwp": 220.0,
            "trl": 6,
            "overall_risk": 55.0,
            "job_creation": 12.0,
            "community_health_risk": 45.0,
        },
    }

    recs = ranker.rank(pathways)
    assert len(recs) == 2
    # Ensure explanation includes strengths/weaknesses related to social
    explanation = recs[0].explanation
    assert explanation is not None


def test_reverse_design_optimizer():
    # Simple evaluator: quadratic relation
    def mock_evaluator(params: dict) -> dict:
        x = params["x"]
        y = params["y"]
        return {
            "gwp": float(x**2 + y**2),
            "npv": float(100.0 - (x - 2.0) ** 2 - (y - 3.0) ** 2),
        }

    parameter_bounds = {
        "x": (0.0, 5.0),
        "y": (0.0, 5.0),
    }

    target_constraints = {
        "gwp": {"type": "max", "value": 25.0},
        "npv": {"type": "min", "value": 85.0},
    }

    optimizer = ReverseDesignOptimizer(
        evaluator_fn=mock_evaluator,
        parameter_bounds=parameter_bounds,
        target_constraints=target_constraints,
        seed=123,
    )

    result = optimizer.run(n_iterations=10, n_initial_points=5)

    assert "best_parameters" in result
    assert "best_metrics" in result
    assert "best_utility" in result
    assert "parameter_sensitivities" in result

    # Check that sensitivities are calculated
    assert "x" in result["parameter_sensitivities"]
    assert "y" in result["parameter_sensitivities"]


def test_benefit_compensation_model():
    model = BenefitCompensationModel()

    baseline_impacts = {
        "climate_change": 500.0,
        "acidification": 50.0,
        "human_toxicity_cancer": 0.05,
    }

    pathway_impacts = {
        "climate_change": 150.0,  # Avoids 350 kg CO2-eq -> $35 benefit
        "acidification": 10.0,  # Avoids 40 mol H+-eq -> $100 benefit
        "human_toxicity_cancer": 0.01,  # Avoids 0.04 -> $200 benefit
    }

    benefits = model.calculate_avoided_damage(baseline_impacts, pathway_impacts)
    total_benefit = benefits["total_avoided_environmental_benefit"]
    assert total_benefit > 0.0

    # Test compensation optimization for a deficit pathway
    comp = model.optimize_compensation(
        pathway_code="PG-SulfurAcid",
        clcc=60.0,
        revenue=45.0,
        avoided_environmental_benefit=total_benefit,
        target_margin=5.0,
    )

    assert comp["pathway_code"] == "PG-SulfurAcid"
    assert comp["required_total_compensation"] > 0.0
    assert "suggested_tipping_fee" in comp
    assert "suggested_subsidy" in comp
