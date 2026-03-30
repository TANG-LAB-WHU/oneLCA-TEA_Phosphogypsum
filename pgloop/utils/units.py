"""
Unit Conversion Module

Standard unit conversions for mass, energy, volume, and temperature.
"""

from dataclasses import dataclass

# Mass conversion factors (to kg)
MASS_TO_KG = {
    "kg": 1.0,
    "g": 0.001,
    "mg": 1e-6,
    "t": 1000.0,  # metric tonne
    "Mt": 1e9,  # megatonne
    "lb": 0.453592,
    "oz": 0.0283495,
    "ton": 907.185,  # US short ton
    "long_ton": 1016.05,  # UK long ton
}

# Energy conversion factors (to MJ)
ENERGY_TO_MJ = {
    "MJ": 1.0,
    "kJ": 0.001,
    "J": 1e-6,
    "GJ": 1000.0,
    "kWh": 3.6,
    "Wh": 0.0036,
    "BTU": 0.001055,
    "MMBTU": 1055.0,
    "therm": 105.5,
    "kcal": 0.004184,
    "cal": 4.184e-6,
    "toe": 41868.0,  # tonne oil equivalent
}

# Volume conversion factors (to m3)
VOLUME_TO_M3 = {
    "m3": 1.0,
    "L": 0.001,
    "mL": 1e-6,
    "cm3": 1e-6,
    "gal": 0.00378541,  # US gallon
    "ft3": 0.0283168,
    "bbl": 0.158987,  # oil barrel
}

# Area conversion factors (to m2)
AREA_TO_M2 = {
    "m2": 1.0,
    "cm2": 1e-4,
    "mm2": 1e-6,
    "km2": 1e6,
    "ha": 10000.0,
    "acre": 4046.86,
    "ft2": 0.092903,
}

# Pressure conversion factors (to Pa)
PRESSURE_TO_PA = {
    "Pa": 1.0,
    "kPa": 1000.0,
    "MPa": 1e6,
    "bar": 1e5,
    "atm": 101325.0,
    "psi": 6894.76,
    "mmHg": 133.322,
    "torr": 133.322,
}


def convert_mass(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert mass between units.

    Args:
        value: Value to convert
        from_unit: Source unit (kg, g, t, lb, etc.)
        to_unit: Target unit

    Returns:
        Converted value
    """
    if from_unit not in MASS_TO_KG:
        raise ValueError(f"Unknown mass unit: {from_unit}")
    if to_unit not in MASS_TO_KG:
        raise ValueError(f"Unknown mass unit: {to_unit}")

    kg_value = value * MASS_TO_KG[from_unit]
    return kg_value / MASS_TO_KG[to_unit]


def convert_energy(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert energy between units.

    Args:
        value: Value to convert
        from_unit: Source unit (MJ, kWh, BTU, etc.)
        to_unit: Target unit

    Returns:
        Converted value
    """
    if from_unit not in ENERGY_TO_MJ:
        raise ValueError(f"Unknown energy unit: {from_unit}")
    if to_unit not in ENERGY_TO_MJ:
        raise ValueError(f"Unknown energy unit: {to_unit}")

    mj_value = value * ENERGY_TO_MJ[from_unit]
    return mj_value / ENERGY_TO_MJ[to_unit]


def convert_volume(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert volume between units.

    Args:
        value: Value to convert
        from_unit: Source unit (m3, L, gal, etc.)
        to_unit: Target unit

    Returns:
        Converted value
    """
    if from_unit not in VOLUME_TO_M3:
        raise ValueError(f"Unknown volume unit: {from_unit}")
    if to_unit not in VOLUME_TO_M3:
        raise ValueError(f"Unknown volume unit: {to_unit}")

    m3_value = value * VOLUME_TO_M3[from_unit]
    return m3_value / VOLUME_TO_M3[to_unit]


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert temperature between units.

    Args:
        value: Temperature value
        from_unit: Source unit (C, K, F)
        to_unit: Target unit

    Returns:
        Converted temperature
    """
    # Convert to Kelvin first
    if from_unit == "K":
        kelvin = value
    elif from_unit == "C":
        kelvin = value + 273.15
    elif from_unit == "F":
        kelvin = (value - 32) * 5 / 9 + 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")

    # Convert from Kelvin
    if to_unit == "K":
        return kelvin
    elif to_unit == "C":
        return kelvin - 273.15
    elif to_unit == "F":
        return (kelvin - 273.15) * 9 / 5 + 32
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")


def convert_pressure(value: float, from_unit: str, to_unit: str) -> float:
    """Convert pressure between units."""
    if from_unit not in PRESSURE_TO_PA:
        raise ValueError(f"Unknown pressure unit: {from_unit}")
    if to_unit not in PRESSURE_TO_PA:
        raise ValueError(f"Unknown pressure unit: {to_unit}")

    pa_value = value * PRESSURE_TO_PA[from_unit]
    return pa_value / PRESSURE_TO_PA[to_unit]


@dataclass
class UnitConverter:
    """
    Unified unit converter with automatic unit detection.
    """

    def convert(self, value: float, from_unit: str, to_unit: str, unit_type: str = None) -> float:
        """
        Convert value between units.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit
            unit_type: Optional type hint (mass, energy, volume, temperature)

        Returns:
            Converted value
        """
        # Auto-detect unit type if not specified
        if unit_type is None:
            unit_type = self._detect_unit_type(from_unit)

        if unit_type == "mass":
            return convert_mass(value, from_unit, to_unit)
        elif unit_type == "energy":
            return convert_energy(value, from_unit, to_unit)
        elif unit_type == "volume":
            return convert_volume(value, from_unit, to_unit)
        elif unit_type == "temperature":
            return convert_temperature(value, from_unit, to_unit)
        elif unit_type == "pressure":
            return convert_pressure(value, from_unit, to_unit)
        else:
            raise ValueError(f"Unknown unit type: {unit_type}")

    def _detect_unit_type(self, unit: str) -> str:
        """Detect unit type from unit string."""
        if unit in MASS_TO_KG:
            return "mass"
        elif unit in ENERGY_TO_MJ:
            return "energy"
        elif unit in VOLUME_TO_M3:
            return "volume"
        elif unit in ("K", "C", "F"):
            return "temperature"
        elif unit in PRESSURE_TO_PA:
            return "pressure"
        else:
            raise ValueError(f"Cannot detect unit type for: {unit}")
