"""
Acid Chemicals

Common acids used in phosphogypsum treatment.
"""

from pgloop.chemicals.base_chemical import Chemical

# Acid chemicals for easy import
ACIDS = {
    "H2SO4": Chemical(
        name="Sulfuric Acid",
        formula="H2SO4",
        cas_number="7664-93-9",
        smiles="OS(=O)(=O)O",
        molecular_weight=98.079,
        density_kg_m3=1840,
        gwp_kg_co2_per_kg=0.09,
        price_usd_per_kg=0.07,
        hazard_class="Corrosive",
    ),
    "HCl": Chemical(
        name="Hydrochloric Acid",
        formula="HCl",
        cas_number="7647-01-0",
        smiles="Cl",
        molecular_weight=36.46,
        density_kg_m3=1180,
        gwp_kg_co2_per_kg=0.8,
        price_usd_per_kg=0.15,
        hazard_class="Corrosive",
    ),
    "HNO3": Chemical(
        name="Nitric Acid",
        formula="HNO3",
        cas_number="7697-37-2",
        smiles="[N+](=O)(O)[O-]",
        molecular_weight=63.01,
        density_kg_m3=1510,
        gwp_kg_co2_per_kg=1.8,
        price_usd_per_kg=0.30,
        hazard_class="Corrosive, Oxidizer",
    ),
    "H3PO4": Chemical(
        name="Phosphoric Acid",
        formula="H3PO4",
        cas_number="7664-38-2",
        smiles="OP(=O)(O)O",
        molecular_weight=97.99,
        density_kg_m3=1880,
        gwp_kg_co2_per_kg=1.2,
        price_usd_per_kg=0.50,
        hazard_class="Corrosive",
    ),
}
