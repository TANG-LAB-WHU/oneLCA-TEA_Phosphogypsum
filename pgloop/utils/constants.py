"""
Constants Module

Physical constants, atomic masses, and LCA reference values.
"""

# Physical constants
PHYSICAL_CONSTANTS = {
    # Fundamental constants
    "R": 8.314,  # J/(mol·K) - Universal gas constant
    "Na": 6.022e23,  # /mol - Avogadro's number
    "k_B": 1.381e-23,  # J/K - Boltzmann constant
    "h": 6.626e-34,  # J·s - Planck constant
    "c": 2.998e8,  # m/s - Speed of light
    "e": 1.602e-19,  # C - Elementary charge
    # Thermodynamic
    "T_ref": 298.15,  # K - Standard temperature
    "P_ref": 101325,  # Pa - Standard pressure
    # Conversion
    "cal_to_J": 4.184,
    "BTU_to_J": 1055.06,
    "atm_to_Pa": 101325,
    # Environmental
    "CO2_C_ratio": 44.01 / 12.01,  # CO2 per C
    "CH4_C_ratio": 16.04 / 12.01,  # CH4 per C
}


# Atomic masses (g/mol)
ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.086,
    "P": 30.974,
    "S": 32.065,
    "Cl": 35.453,
    "K": 39.098,
    "Ca": 40.078,
    "Ti": 47.867,
    "Fe": 55.845,
    "Cu": 63.546,
    "Zn": 65.38,
    "As": 74.922,
    "Sr": 87.62,
    "Cd": 112.41,
    "Ba": 137.33,
    "Pb": 207.2,
    "U": 238.03,
    "Ra": 226.0,
    # Rare Earth Elements
    "La": 138.91,
    "Ce": 140.12,
    "Pr": 140.91,
    "Nd": 144.24,
    "Sm": 150.36,
    "Eu": 151.96,
    "Gd": 157.25,
    "Tb": 158.93,
    "Dy": 162.50,
    "Ho": 164.93,
    "Er": 167.26,
    "Tm": 168.93,
    "Yb": 173.05,
    "Lu": 174.97,
    "Y": 88.906,
    "Sc": 44.956,
}


# LCA normalization reference values (per capita per year)
LCA_REFERENCE_VALUES = {
    # EU27 2010 values (from JRC)
    "EU27": {
        "climate_change_kg_co2": 9220,
        "acidification_mol_h": 47.6,
        "eutrophication_fresh_kg_p": 1.58,
        "eutrophication_marine_kg_n": 19.5,
        "human_toxicity_cancer_ctuh": 1.69e-5,
        "human_toxicity_noncancer_ctuh": 1.34e-4,
        "ecotoxicity_fresh_ctue": 9100,
        "ionizing_radiation_kbq_u235": 1040,
        "particulate_matter_disease": 6.26e-4,
        "resource_depletion_kg_sb": 0.0635,
    },
    # World average
    "World": {
        "climate_change_kg_co2": 4800,
        "acidification_mol_h": 38.0,
        "eutrophication_fresh_kg_p": 1.2,
        "particulate_matter_disease": 8.0e-4,
    },
}


# Emission factors
EMISSION_FACTORS = {
    # Fuel combustion (kg CO2 per unit)
    "diesel_kg_co2_per_kg": 3.17,
    "diesel_kg_co2_per_L": 2.68,
    "natural_gas_kg_co2_per_mj": 0.055,
    "natural_gas_kg_co2_per_m3": 1.93,
    "coal_kg_co2_per_kg": 2.42,
    "lpg_kg_co2_per_kg": 3.03,
    # Electricity (kg CO2 per kWh)
    "electricity_world_avg": 0.475,
    "electricity_us": 0.386,
    "electricity_eu": 0.276,
    "electricity_china": 0.555,
    "electricity_india": 0.708,
    "electricity_brazil": 0.074,
}


# Shadow prices for external costs (USD per unit, 2024)
SHADOW_PRICES = {
    "co2_usd_per_t": 100.0,  # Social cost of carbon
    "so2_usd_per_kg": 5.0,
    "nox_usd_per_kg": 8.0,
    "pm25_usd_per_kg": 30.0,
    "pm10_usd_per_kg": 15.0,
    "nh3_usd_per_kg": 12.0,
    "voc_usd_per_kg": 3.0,
    "phosphorus_usd_per_kg": 25.0,
    "nitrogen_usd_per_kg": 10.0,
}


def get_molecular_weight(formula: str) -> float:
    """
    Calculate molecular weight from formula.

    Simple parser for formulas like H2O, H2SO4, Ca(OH)2.
    """
    import re

    # Handle parentheses by expansion (simplified)
    # This is a basic implementation
    total = 0.0

    # Pattern: element followed by optional number
    pattern = r"([A-Z][a-z]?)(\d*)"
    matches = re.findall(pattern, formula)

    for element, count in matches:
        if element and element in ATOMIC_MASSES:
            n = int(count) if count else 1
            total += ATOMIC_MASSES[element] * n

    return total
