"""
PG-LCA-TEA Test Suite
"""

import pytest


def test_import_modules():
    """Test that all main modules can be imported."""
    from pgloop import LCAEngine, TEAEngine
    from pgloop.iodata import PDFParser, DataStandardizer, APIConnector
    from pgloop.knowledge import PhosphogypsumKG, LLMExtractor, LightRAGEngine, GapFiller
    from pgloop.lca import LCAEngine, LifeCycleInventory, ImpactAssessment
    from pgloop.tea import TEAEngine, CAPEXCalculator, OPEXCalculator
    from pgloop.pathways import (
        StackDisposalPathway,
        CementPathway,
        get_pathway,
        list_pathways
    )
    
    assert True


def test_pathway_registry():
    """Test pathway registry functions."""
    from pgloop.pathways import list_pathways, get_pathway
    
    pathways = list_pathways()
    assert len(pathways) == 6
    assert "PG-Stack" in pathways
    assert "PG-CementProd" in pathways


def test_stack_disposal_pathway():
    """Test stack disposal pathway initialization."""
    from pgloop.pathways import StackDisposalPathway
    
    pathway = StackDisposalPathway(country="China")
    assert pathway.code == "PG-Stack"
    assert pathway.name == "Stack Disposal"
    assert pathway.trl == 9


def test_lifecycle_inventory():
    """Test LCI creation and scaling."""
    from pgloop.lca.inventory import LifeCycleInventory
    
    lci = LifeCycleInventory(
        process_name="Test Process",
        functional_unit="1 kg",
        functional_unit_value=1.0
    )
    
    lci.add_input("Material A", 1.0, "kg")
    lci.add_emission("CO2", 0.5, "kg", "air")
    
    scaled = lci.scale_to(1000)
    
    assert scaled.functional_unit_value == 1000
    assert scaled.inputs[0].quantity == 1000
    assert scaled.emissions_air[0].quantity == 500


def test_characterization_factors():
    """Test characterization factors retrieval."""
    from pgloop.lca.characterization import CharacterizationFactors
    
    cf = CharacterizationFactors()
    
    co2_gwp = cf.get_factor("climate_change", "co2")
    assert co2_gwp == 1.0
    
    ch4_gwp = cf.get_factor("climate_change", "ch4")
    assert ch4_gwp == 28.0


def test_capex_calculation():
    """Test CAPEX calculation."""
    from pgloop.tea.capex import CAPEXCalculator
    
    calc = CAPEXCalculator({"discount_rate": 0.05, "lifetime_years": 20})
    
    capex_data = {
        "equipment": [
            {"name": "Reactor", "cost": 100000}
        ]
    }
    
    total = calc.calculate_total(capex_data)
    assert total > 100000  # Should include installation factors


def test_monte_carlo():
    """Test Monte Carlo simulation."""
    from pgloop.uncertainty.direct_sampling import MonteCarloSimulator
    
    mc = MonteCarloSimulator(n_iterations=100, seed=42)
    
    samples = mc.sample_triangular(80, 100, 120)
    assert len(samples) == 100
    assert samples.mean() > 80
    assert samples.mean() < 120


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
