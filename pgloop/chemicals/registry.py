"""
Chemical Registry

Database of common chemicals with properties and LCA data.
"""

from typing import Dict, List, Optional
from pgloop.chemicals.base_chemical import Chemical


# Main chemical database
CHEMICAL_DATABASE: Dict[str, Chemical] = {
    # === Acids ===
    "H2SO4": Chemical(
        name="Sulfuric Acid",
        formula="H2SO4",
        cas_number="7664-93-9",
        smiles="OS(=O)(=O)O",
        molecular_weight=98.079,
        density_kg_m3=1840,
        state="liquid",
        gwp_kg_co2_per_kg=0.09,
        acidification_kg_so2_per_kg=0.02,
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
        state="liquid",
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
        state="liquid",
        gwp_kg_co2_per_kg=1.8,
        acidification_kg_so2_per_kg=0.05,
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
        state="liquid",
        gwp_kg_co2_per_kg=1.2,
        eutrophication_kg_po4_per_kg=0.1,
        price_usd_per_kg=0.50,
        hazard_class="Corrosive",
    ),
    
    # === Bases ===
    "NaOH": Chemical(
        name="Sodium Hydroxide",
        formula="NaOH",
        cas_number="1310-73-2",
        smiles="[Na+].[OH-]",
        molecular_weight=40.0,
        density_kg_m3=2130,
        state="solid",
        gwp_kg_co2_per_kg=1.2,
        price_usd_per_kg=0.40,
        hazard_class="Corrosive",
    ),
    "NH3": Chemical(
        name="Ammonia",
        formula="NH3",
        cas_number="7664-41-7",
        smiles="N",
        molecular_weight=17.03,
        density_kg_m3=682,
        boiling_point_k=239.8,
        state="gas",
        gwp_kg_co2_per_kg=2.1,
        acidification_kg_so2_per_kg=0.001,
        price_usd_per_kg=0.35,
        hazard_class="Toxic, Corrosive",
    ),
    "CaO": Chemical(
        name="Calcium Oxide (Quicklime)",
        formula="CaO",
        cas_number="1305-78-8",
        smiles="[Ca]=O",
        molecular_weight=56.08,
        density_kg_m3=3340,
        state="solid",
        gwp_kg_co2_per_kg=0.95,
        price_usd_per_kg=0.08,
        hazard_class="Corrosive",
    ),
    "Ca(OH)2": Chemical(
        name="Calcium Hydroxide (Slaked Lime)",
        formula="Ca(OH)2",
        cas_number="1305-62-0",
        smiles="[Ca+2].[OH-].[OH-]",
        molecular_weight=74.09,
        density_kg_m3=2211,
        state="solid",
        gwp_kg_co2_per_kg=0.75,
        price_usd_per_kg=0.10,
        hazard_class="Irritant",
    ),
    "(NH4)2CO3": Chemical(
        name="Ammonium Carbonate",
        formula="(NH4)2CO3",
        cas_number="506-87-6",
        smiles="[NH4+].[NH4+].[O-]C([O-])=O",
        molecular_weight=96.09,
        density_kg_m3=1500,
        state="solid",
        gwp_kg_co2_per_kg=1.5,
        price_usd_per_kg=0.60,
        hazard_class="Irritant",
    ),
    
    # === Solvents/Extractants ===
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
    
    # === Common Reagents ===
    "CO2": Chemical(
        name="Carbon Dioxide",
        formula="CO2",
        cas_number="124-38-9",
        smiles="O=C=O",
        molecular_weight=44.01,
        density_kg_m3=1.98,  # Gas at STP
        state="gas",
        gwp_kg_co2_per_kg=1.0,
        price_usd_per_kg=0.05,
        hazard_class="Asphyxiant",
    ),
    "H2O": Chemical(
        name="Water",
        formula="H2O",
        cas_number="7732-18-5",
        smiles="O",
        molecular_weight=18.015,
        density_kg_m3=998,
        boiling_point_k=373.15,
        melting_point_k=273.15,
        state="liquid",
        gwp_kg_co2_per_kg=0.001,
        price_usd_per_kg=0.002,
        hazard_class="None",
    ),
}


def get_chemical(identifier: str) -> Optional[Chemical]:
    """
    Get chemical by name, formula, or CAS number.
    
    Args:
        identifier: Chemical name, formula, or CAS number
        
    Returns:
        Chemical object or None if not found
    """
    # Direct lookup by formula
    if identifier in CHEMICAL_DATABASE:
        return CHEMICAL_DATABASE[identifier]
    
    # Search by name or CAS
    identifier_lower = identifier.lower()
    for chem in CHEMICAL_DATABASE.values():
        if chem.name.lower() == identifier_lower:
            return chem
        if chem.cas_number == identifier:
            return chem
    
    return None


def list_chemicals() -> List[str]:
    """List all chemical formulas in database."""
    return list(CHEMICAL_DATABASE.keys())


def add_chemical(chemical: Chemical) -> None:
    """Add a chemical to the database."""
    CHEMICAL_DATABASE[chemical.formula] = chemical
