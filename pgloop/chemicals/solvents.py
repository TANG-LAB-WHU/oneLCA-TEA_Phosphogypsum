"""
Solvent Chemicals

Organic solvents and extractants for REE recovery.
"""

from pgloop.chemicals.base_chemical import Chemical

# Solvent chemicals for easy import
SOLVENTS = {
    "D2EHPA": Chemical(
        name="Di-(2-ethylhexyl)phosphoric Acid",
        formula="C16H35O4P",
        cas_number="298-07-7",
        smiles="CCCCC(CC)COP(=O)(O)OCC(CC)CCCC",
        molecular_weight=322.42,
        density_kg_m3=970,
        state="liquid",
        gwp_kg_co2_per_kg=3.5,
        price_usd_per_kg=5.00,
        hazard_class="Irritant",
    ),
    "TBP": Chemical(
        name="Tributyl Phosphate",
        formula="C12H27O4P",
        cas_number="126-73-8",
        smiles="CCCCOP(=O)(OCCCC)OCCCC",
        molecular_weight=266.31,
        density_kg_m3=979,
        state="liquid",
        gwp_kg_co2_per_kg=4.0,
        price_usd_per_kg=3.50,
        hazard_class="Irritant",
    ),
    "Kerosene": Chemical(
        name="Kerosene (Diluent)",
        formula="C12H26",
        cas_number="8008-20-6",
        smiles="CCCCCCCCCCCC",
        molecular_weight=170.34,
        density_kg_m3=800,
        boiling_point_k=473,
        state="liquid",
        gwp_kg_co2_per_kg=0.5,
        price_usd_per_kg=0.80,
        hazard_class="Flammable",
    ),
    "EHEHPA": Chemical(
        name="2-Ethylhexyl Phosphonic Acid Mono-2-Ethylhexyl Ester",
        formula="C16H35O3P",
        cas_number="14802-03-0",
        smiles="CCCCC(CC)CP(=O)(O)OCC(CC)CCCC",
        molecular_weight=306.42,
        density_kg_m3=960,
        state="liquid",
        gwp_kg_co2_per_kg=4.0,
        price_usd_per_kg=8.00,
        hazard_class="Irritant",
    ),
    "Cyanex272": Chemical(
        name="Cyanex 272",
        formula="C16H35O2PS",
        cas_number="24570-51-0",
        smiles="CCCCC(CC)CP(=S)(O)OCC(CC)CCCC",
        molecular_weight=322.48,
        density_kg_m3=920,
        state="liquid",
        gwp_kg_co2_per_kg=5.0,
        price_usd_per_kg=15.00,
        hazard_class="Irritant",
    ),
}
