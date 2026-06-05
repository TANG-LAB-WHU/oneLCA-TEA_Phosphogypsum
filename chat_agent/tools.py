"""
Provides wrapper functions that expose pgloop backend functionalities (LCA, TEA, RAG) 
as standardized tools for the LLM function calling interface.
"""

import json
from typing import Dict, Any

from pgloop import LCAEngine, PathwayRanker, RiskAggregator, TEAEngine, get_pathway, list_pathways
from pgloop.decision.scenario import MARKET_SCENARIOS, ScenarioAnalyzer
from pgloop.knowledge.lightrag_engine import LightRAGEngine, LIGHTRAG_AVAILABLE
from pgloop.risk.aggregator import RiskScore
from pgloop.utils.currency import format_currency

# Initialize engines globally to reuse them across tool calls
lca_engine = LCAEngine()
tea_engine = TEAEngine(country="China")
risk_aggregator = RiskAggregator()
pathway_ranker = PathwayRanker()

if LIGHTRAG_AVAILABLE:
    rag_engine = LightRAGEngine()
else:
    rag_engine = None

def get_available_pathways() -> str:
    """
    Get a list of all available phosphogypsum valorization pathways.
    Call this tool to understand what pathway codes are valid for analysis.
    
    Returns:
        A JSON string containing the available pathway codes and names.
    """
    pathways = list_pathways()
    return json.dumps(pathways, indent=2)

def search_literature(query: str, mode: str = "hybrid") -> str:
    """
    Search the phosphogypsum scientific literature using GraphRAG (LightRAG).
    Use this to retrieve factual knowledge, parameters, reaction kinetics, or contextual information from papers.
    
    Args:
        query: The specific question or search term.
        mode: The query mode (default: "hybrid", other options: "local", "global", "mix").
        
    Returns:
        The text answer containing retrieved facts and citations.
    """
    if not rag_engine:
        return "Error: LightRAG is not available. Please install lightrag-hku."
    
    try:
        print(f"\n[Tool Execution] Searching literature for: '{query}'...")
        result = rag_engine.query(query, mode=mode)
        return result.answer
    except Exception as e:
        return f"Error executing literature search: {str(e)}"

def calculate_lca_tea(pathway_code: str) -> str:
    """
    Run Life Cycle Assessment (LCA) and Techno-Economic Analysis (TEA) for a given phosphogypsum pathway.
    
    Args:
        pathway_code: The code of the pathway to analyze (e.g., "PG-CementProd", "PG-REEextract", "PG-Stack").
        
    Returns:
        A JSON string containing the environmental impacts (e.g., GWP, Resource Depletion) and economic metrics (e.g., NPV, CAPEX, OPEX).
    """
    try:
        print(f"\n[Tool Execution] Calculating LCA/TEA for pathway: '{pathway_code}'...")
        pathway = get_pathway(pathway_code)
        
        lca_result = lca_engine.calculate(pathway, functional_unit_value=1.0)
        tea_result = tea_engine.calculate(pathway, functional_unit_value=1.0)
        npv_result = tea_engine.calculate_npv(pathway)
        
        result_dict = {
            "pathway_code": pathway_code,
            "pathway_name": pathway.name,
            "LCA_Impacts": lca_result.impacts,
            "TEA_Metrics": {
                "CAPEX": tea_result.capex,
                "OPEX": tea_result.opex,
                "Revenues": tea_result.revenues,
                "CLCC": tea_result.clcc,
                "NPV": npv_result.get("npv", 0),
                "IRR": npv_result.get("irr", 0),
                "Payback_Years": npv_result.get("payback_years", 0)
            }
        }
        return json.dumps(result_dict, indent=2)
    except Exception as e:
        return f"Error calculating LCA/TEA: {str(e)}"

def rank_all_pathways() -> str:
    """
    Perform a Multi-Criteria Decision Analysis (MCDA) to evaluate and rank all available phosphogypsum pathways.
    
    Returns:
        A formatted string with the Pareto optimal recommendations and scores.
    """
    try:
        print(f"\n[Tool Execution] Ranking all pathways...")
        pathway_codes = ["PG-Stack", "PG-CementProd", "PG-REEextract"]
        decision_data = {}
        
        for code in pathway_codes:
            pathway = get_pathway(code)
            lca_result = lca_engine.calculate(pathway, functional_unit_value=1.0)
            npv_result = tea_engine.calculate_npv(pathway)
            
            decision_data[pathway.name] = {
                "gwp": lca_result.impacts.get("climate_change", 0),
                "resource_depletion": lca_result.impacts.get("resource_depletion", 0),
                "human_toxicity": lca_result.impacts.get("human_toxicity", 0),
                "npv": npv_result.get("npv", 0) / 1000000,
                "irr": npv_result.get("irr", 0),
                "payback": npv_result.get("payback_years", 20),
                "trl": pathway.trl,
                "scalability": 0.8,
                "overall_risk": 50.0  # Placeholder for full risk aggregation
            }
            
        recommendations = pathway_ranker.rank(decision_data)
        
        result_str = "Pathway Rankings:\n"
        for rec in recommendations:
            status = " [OPTIMAL]" if rec.is_pareto_optimal else ""
            result_str += f"- Rank {rec.rank}: {rec.pathway_name}{status} (Score: {rec.score:.3f})\n"
            result_str += f"  Explanation: {rec.explanation}\n"
            
        return result_str
    except Exception as e:
        return f"Error ranking pathways: {str(e)}"

def run_market_robustness_scenario(pathway_code: str) -> str:
    """
    Run a market robustness scenario analysis for a specific pathway to see how it performs under different economic conditions.
    
    Args:
        pathway_code: The code of the pathway to analyze.
        
    Returns:
        A JSON string containing the Cost (CLCC) under Baseline, Optimistic, and Pessimistic scenarios.
    """
    try:
        print(f"\n[Tool Execution] Running market robustness for: '{pathway_code}'...")
        pathway = get_pathway(pathway_code)
        analyzer = ScenarioAnalyzer(lca_engine, tea_engine)
        
        robustness = analyzer.quick_robustness_check(
            pathway=pathway,
            scenarios=list(MARKET_SCENARIOS.values())[:3],
            metric="clcc",
        )
        
        return json.dumps(robustness, indent=2)
    except Exception as e:
        return f"Error running scenario analysis: {str(e)}"

# Define the list of tools that will be provided to the agent
AVAILABLE_TOOLS = {
    "get_available_pathways": get_available_pathways,
    "search_literature": search_literature,
    "calculate_lca_tea": calculate_lca_tea,
    "rank_all_pathways": rank_all_pathways,
    "run_market_robustness_scenario": run_market_robustness_scenario
}
