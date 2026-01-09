"""
Base Chemicals

Common bases and alkalis used in phosphogypsum treatment.
"""

from pgloop.chemicals.base_chemical import Chemical

# Base chemicals for easy import
BASES = {
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
}
