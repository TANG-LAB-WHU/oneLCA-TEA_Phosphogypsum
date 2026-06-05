"""
Basic run example for PhosphogypsumBot.

This script demonstrates how to initialize the LCA and TEA engines,
load a predefined valorization pathway, and calculate environmental
and techno-economic footprints.
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pgloop import LCAEngine, TEAEngine, get_pathway
from pgloop.utils.currency import format_currency


def main():
    print("=" * 60)
    print("   PhosphogypsumBot: Basic Pathway Assessment Example   ")
    print("=" * 60)

    # 1. Initialize evaluation engines
    lca_engine = LCAEngine()
    tea_engine = TEAEngine(country="China")

    # 2. Load the Sulfuric Acid and Cement Co-production Pathway (B1)
    pathway_code = "PG-SulfurAcid"
    print(f"\nLoading pathway: {pathway_code}...")
    pathway = get_pathway(pathway_code)

    print(f"Name: {pathway.name}")
    # Display pathway default parameters
    print("Parameters:")
    for param, value in pathway.parameters.items():
        print(f"  - {param}: {value}")

    # 3. Calculate Life Cycle Assessment (LCA) environmental impacts
    print("\n[LCA] Calculating environmental impacts...")
    lca_result = lca_engine.calculate(pathway, functional_unit_value=1.0)

    print("Environmental Indicators (per 1 tonne PG treated):")
    for category, value in lca_result.impacts.items():
        unit = lca_engine.get_indicator_unit(category) or "kg-eq"
        print(f"  - {category:25}: {value:.4f} {unit}")

    # 4. Calculate Techno-Economic Analysis (TEA) financial metrics
    print("\n[TEA] Calculating economic costs...")
    tea_result = tea_engine.calculate(pathway, functional_unit_value=1.0)
    npv_result = tea_engine.calculate_npv(pathway)

    print("Financial Performance Indicators:")
    print(f"  - Conventional Cost (CLCC) : {format_currency(tea_result.clcc)} / t PG")
    print(f"  - Societal Cost (SLCC)     : {format_currency(tea_result.slcc)} / t PG")
    print(f"  - Net Present Value (NPV)  : {format_currency(npv_result['npv'])}")
    print(f"  - Internal Rate of Return  : {npv_result['irr'] * 100:.2f}%")
    print(f"  - Payback Period           : {npv_result['payback_years']:.1f} years")

    print("\n" + "=" * 60)
    print("Assessment Completed Successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
