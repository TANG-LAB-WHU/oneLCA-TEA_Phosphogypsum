import json
from unittest.mock import MagicMock, patch

import pytest

from chat_agent.agent import PhosphogypsumAgent, function_to_schema
from chat_agent.tools import (
    AVAILABLE_TOOLS,
    calibrate_process_parameters,
    calculate_lca_tea,
    get_available_pathways,
    optimize_benefit_compensation,
    optimize_reverse_design,
    predict_crystal_properties,
    query_realtime_telemetry,
    rank_all_pathways,
    run_market_robustness_scenario,
    search_literature,
)


def test_all_10_tools_registered():
    """Verify that all 10 tools are registered in AVAILABLE_TOOLS."""
    expected_tools = {
        "get_available_pathways",
        "search_literature",
        "calculate_lca_tea",
        "rank_all_pathways",
        "run_market_robustness_scenario",
        "optimize_reverse_design",
        "optimize_benefit_compensation",
        "calibrate_process_parameters",
        "predict_crystal_properties",
        "query_realtime_telemetry",
    }
    assert set(AVAILABLE_TOOLS.keys()) == expected_tools


def test_tool_schemas_generation():
    """Verify that function_to_schema successfully extracts valid OpenAI schemas for all tools."""
    for name, func in AVAILABLE_TOOLS.items():
        schema = function_to_schema(func)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == name
        assert len(schema["function"]["description"]) > 0
        assert "properties" in schema["function"]["parameters"]


def test_optimize_reverse_design_tool():
    """Test Bayesian reverse design optimization tool execution."""
    raw_res = optimize_reverse_design(
        pathway_code="PG-SulfurAcid",
        target_gwp_max=150.0,
        target_npv_min=10.0,
        n_iterations=5,
    )
    res = json.loads(raw_res)
    assert res["pathway_code"] == "PG-SulfurAcid"
    assert "optimal_parameters" in res
    assert "temperature_c" in res["optimal_parameters"]
    assert "parameter_sensitivities" in res


def test_optimize_benefit_compensation_tool():
    """Test Benefit Compensation model tool execution."""
    raw_res = optimize_benefit_compensation(
        pathway_code="PG-CementProd",
        target_margin=5.0,
    )
    res = json.loads(raw_res)
    assert "pathway_code" in res
    assert "required_total_compensation" in res
    assert "suggested_tipping_fee" in res
    assert "suggested_subsidy" in res


def test_calibrate_process_parameters_tool():
    """Test MCMC parameter calibration tool execution."""
    raw_res = calibrate_process_parameters(
        pathway_code="PG-CementProd",
        n_samples=200,
    )
    res = json.loads(raw_res)
    assert res["pathway_code"] == "PG-CementProd"
    assert "posterior_means" in res
    assert "moisture_fraction" in res["posterior_means"]
    assert "95_credible_intervals" in res


def test_predict_crystal_properties_tool():
    """Test crystal property prediction tool execution."""
    raw_res = predict_crystal_properties("gypsum")
    res = json.loads(raw_res)
    assert "formula" in res or "name" in res
    assert "density_kg_m3" in res or "density" in res


def test_query_realtime_telemetry_tool():
    """Test IoT telemetry query tool execution."""
    raw_res = query_realtime_telemetry(metric_name="all", limit=3)
    res = json.loads(raw_res)
    assert "records" in res
    assert len(res["records"]) > 0


def test_calculate_lca_tea_tool():
    """Test forward LCA-TEA calculation tool."""
    raw_res = calculate_lca_tea("PG-CementProd")
    res = json.loads(raw_res)
    assert res["pathway_code"] == "PG-CementProd"
    assert "LCA_Impacts" in res
    assert "TEA_Metrics" in res


def test_rank_all_pathways_tool():
    """Test 5D TEPES multi-criteria ranking tool."""
    res_str = rank_all_pathways()
    assert "Pathway Rankings" in res_str
    assert "Score:" in res_str


def test_get_available_pathways_tool():
    """Test pathway listing tool."""
    raw_res = get_available_pathways()
    res = json.loads(raw_res)
    assert len(res) > 0


def test_search_literature_tool():
    """Test literature search tool."""
    res = search_literature("What is the composition of phosphogypsum?")
    assert len(res) > 0
    assert "CaSO4" in res or "Phosphogypsum" in res


@patch("chat_agent.agent.OpenAI")
def test_agent_multi_step_tool_calling(mock_openai_class):
    """Test that the agent can execute multi-step tool calls in sequence."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    # Step 1: Agent calls reverse design tool
    tool_call_1 = MagicMock()
    tool_call_1.id = "call_rev_1"
    tool_call_1.function.name = "optimize_reverse_design"
    tool_call_1.function.arguments = json.dumps({"pathway_code": "PG-SulfurAcid", "n_iterations": 3})

    msg_1 = MagicMock()
    msg_1.tool_calls = [tool_call_1]
    msg_1.content = None

    choice_1 = MagicMock()
    choice_1.message = msg_1

    # Step 2: Agent formulates final answer
    msg_2 = MagicMock()
    msg_2.tool_calls = None
    msg_2.content = "Optimal temperature is 1180°C with GWP satisfied."

    choice_2 = MagicMock()
    choice_2.message = msg_2

    resp_1 = MagicMock(choices=[choice_1])
    resp_2 = MagicMock(choices=[choice_2])

    mock_client.chat.completions.create.side_effect = [resp_1, resp_2]

    agent = PhosphogypsumAgent()
    final_output = agent.chat("Optimize sulfuric acid pathway parameters for me.")

    assert "1180°C" in final_output
    assert mock_client.chat.completions.create.call_count == 2
