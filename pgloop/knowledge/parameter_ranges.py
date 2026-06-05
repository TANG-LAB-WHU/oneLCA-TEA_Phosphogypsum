"""
Build pathway-ready parameter ranges from extracted JSON data.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

from pgloop.pathways import PATHWAYS

NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
PER_TONNE_RE = re.compile(r"(?:/|per\s+)(?:tonne|tonnes|ton|t)\b")
PER_KG_RE = re.compile(r"(?:/|per\s+)kg\b")
PER_G_RE = re.compile(r"(?:/|per\s+)g\b")


def _identity(value: float) -> float:
    return value


def _to_kg_from_m3(value: float) -> float:
    # Assume water-like density (1000 kg/m3) for coarse pathway calibration.
    return value * 1000.0


def _caso4_to_ca_available(value: float) -> float:
    # Ca mass fraction in CaSO4 ~= 40.078 / 136.14.
    return value * 0.294


def _caso4_to_s_available(value: float) -> float:
    # S mass fraction in CaSO4 ~= 32.06 / 136.14.
    return value * 0.235


PATHWAY_MAPPING_RULES: dict[str, dict[str, list[tuple[str, Callable[[float], float]]]]] = {
    "PG-CementProd": {
        "drying_energy_mj_per_t": [("steam_mj_per_t", _identity)],
        "electricity_kwh_per_t": [("electricity_kwh_per_t", _identity)],
        "caso4_fraction": [("caso4_fraction", _identity)],
        "moisture_fraction": [("moisture_fraction", _identity)],
        "ra226_bq_kg": [("ra226_bq_kg", _identity)],
        "usable_fraction": [("process_yield_fraction", _identity)],
    },
    "PG-REEextract": {
        "moisture_content": [("moisture_fraction", _identity)],
        "leaching_efficiency": [("process_yield_fraction", _identity)],
        "purification_efficiency": [("process_yield_fraction", _identity)],
        "h2so4_kg_per_t": [("h2so4_kg_per_t", _identity)],
        "sodium_hydroxide_kg_per_t": [("sodium_hydroxide_kg_per_t", _identity)],
        "extractant_l_per_t": [("extractant_l_per_t", _identity)],
        "process_water_m3_per_t": [("process_water_m3_per_t", _identity)],
        "electricity_kwh_per_t": [("electricity_kwh_per_t", _identity)],
        "steam_mj_per_t": [("steam_mj_per_t", _identity)],
        "residue_treatment_cost_usd_t": [("residue_treatment_cost_usd_t", _identity)],
        "ree_market_price_usd_kg": [("ree_market_price_usd_kg", _identity)],
    },
    "PG-Stack": {
        "electricity_kwh_per_t": [("electricity_kwh_per_t", _identity)],
        "caso4_fraction": [("caso4_fraction", _identity)],
        "p2o5_fraction": [("p2o5_fraction", _identity)],
        "f_fraction": [("f_fraction", _identity)],
        "ra226_bq_kg": [("ra226_bq_kg", _identity)],
        "leachate_m3_per_t": [("leachate_m3_per_t", _identity)],
    },
    "PG-ConstructMat": {
        "processing_energy_kwh_per_t": [("electricity_kwh_per_t", _identity)],
        "binder_kg_per_t": [("binder_kg_per_t", _identity)],
        "water_kg_per_t": [("process_water_m3_per_t", _to_kg_from_m3)],
        "product_yield": [("process_yield_fraction", _identity)],
    },
    "PG-Soil": {
        "application_energy_kwh_per_t": [("electricity_kwh_per_t", _identity)],
        "ca_available_fraction": [("caso4_fraction", _caso4_to_ca_available)],
        "s_available_fraction": [("caso4_fraction", _caso4_to_s_available)],
    },
    "PG-ChemReco": {
        "ammonia_kg_per_t": [("ammonia_kg_per_t", _identity)],
        "co2_input_kg_per_t": [("co2_input_kg_per_t", _identity)],
        "energy_kwh_per_t": [("electricity_kwh_per_t", _identity)],
        "ammonium_sulfate_yield": [("process_yield_fraction", _identity)],
        "caco3_yield": [("process_yield_fraction", _identity)],
    },
    "PG-SulfurAcid": {
        "coal_reducing_kg_per_t": [("coal_reducing_kg_per_t", _identity)],
        "coal_heating_kg_per_t": [("coal_heating_kg_per_t", _identity)],
        "additives_kg_per_t": [("binder_kg_per_t", _identity)],
        "electricity_kwh_per_t": [("electricity_kwh_per_t", _identity)],
        "sulfuric_acid_yield": [("process_yield_fraction", _identity)],
        "sulfur_yield": [("process_yield_fraction", _identity)],
        "clinker_yield": [("process_yield_fraction", _identity)],
        "moisture_fraction": [("moisture_fraction", _identity)],
    },
}

PARAMETER_UNITS: dict[str, str] = {
    "caso4_fraction": "mass_fraction",
    "p2o5_fraction": "mass_fraction",
    "f_fraction": "mass_fraction",
    "moisture_fraction": "mass_fraction",
    "process_yield_fraction": "mass_fraction",
    "ra226_bq_kg": "Bq/kg",
    "electricity_kwh_per_t": "kWh/t",
    "steam_mj_per_t": "MJ/t",
    "process_water_m3_per_t": "m3/t",
    "h2so4_kg_per_t": "kg/t",
    "sodium_hydroxide_kg_per_t": "kg/t",
    "ammonia_kg_per_t": "kg/t",
    "co2_input_kg_per_t": "kg/t",
    "extractant_l_per_t": "L/t",
    "binder_kg_per_t": "kg/t",
    "leachate_m3_per_t": "m3/t",
    "residue_treatment_cost_usd_t": "USD/t",
    "ree_market_price_usd_kg": "USD/kg",
    "trl": "level",
}


def _lower_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower()
    try:
        return json.dumps(value, ensure_ascii=False).lower()
    except TypeError:
        return str(value).lower()


def _extract_numbers(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        cleaned = value.replace(",", "")
        return [float(match.group(0)) for match in NUMBER_RE.finditer(cleaned)]
    if isinstance(value, dict):
        ordered_keys = ("amount", "value", "mode", "mean", "min", "max")
        preferred: list[float] = []
        for key in ordered_keys:
            if key in value:
                preferred.extend(_extract_numbers(value[key]))
        if preferred:
            return preferred
        nested: list[float] = []
        for nested_value in value.values():
            nested.extend(_extract_numbers(nested_value))
        return nested
    if isinstance(value, (list, tuple)):
        numbers: list[float] = []
        for item in value:
            numbers.extend(_extract_numbers(item))
        return numbers
    return []


def _extract_first_number(value: Any) -> float | None:
    numbers = _extract_numbers(value)
    if not numbers:
        return None
    return numbers[0]


def _get_field(data: dict[str, Any], *names: str) -> Any:
    if not isinstance(data, dict):
        return None
    lowered = {str(key).lower(): key for key in data}
    for name in names:
        if name in data:
            return data[name]
        key = lowered.get(name.lower())
        if key is not None:
            return data[key]
    return None


def _normalize_fraction(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    if value < 0:
        return None
    if value > 1 and value <= 100:
        value /= 100.0
    # Ignore values that are very unlikely to be fractions (e.g., ppm).
    if value > 1.5:
        return None
    return value


def _functional_unit_scale_to_tonne(functional_unit: Any) -> float:
    text = _lower_text(functional_unit)
    quantity = _extract_first_number(functional_unit) or 1.0
    quantity = max(quantity, 1e-12)
    if "tonne" in text or "tons" in text or " ton " in f" {text} " or " t " in f" {text} ":
        return 1.0 / quantity
    if "kg" in text:
        return 1000.0 / quantity
    if re.search(r"(?:^|[^a-z])g(?:[^a-z]|$)", text):
        return 1_000_000.0 / quantity
    return 1.0


def _scale_to_per_tonne(amount: float, unit_text: str, fu_scale: float) -> float:
    if PER_TONNE_RE.search(unit_text):
        return amount
    if PER_KG_RE.search(unit_text):
        return amount * 1000.0
    if PER_G_RE.search(unit_text):
        return amount * 1_000_000.0
    return amount * fu_scale


def _convert_energy_to_kwh(amount: float, unit_text: str) -> float:
    if "mj" in unit_text:
        return amount / 3.6
    if "gj" in unit_text:
        return amount * (1000.0 / 3.6)
    if "wh" in unit_text and "kwh" not in unit_text:
        return amount / 1000.0
    if "j" in unit_text and "mj" not in unit_text and "gj" not in unit_text:
        return amount / 3_600_000.0
    return amount


def _convert_energy_to_mj(amount: float, unit_text: str) -> float:
    if "kwh" in unit_text:
        return amount * 3.6
    if "wh" in unit_text and "kwh" not in unit_text:
        return amount * 0.0036
    if "gj" in unit_text:
        return amount * 1000.0
    if "j" in unit_text and "mj" not in unit_text and "gj" not in unit_text:
        return amount / 1_000_000.0
    return amount


def _convert_mass_to_kg(amount: float, unit_text: str) -> float | None:
    if "mg" in unit_text:
        return amount / 1_000_000.0
    if re.search(r"(?:^|[^a-z])g(?:[^a-z]|$)", unit_text) and "kg" not in unit_text:
        return amount / 1000.0
    if "ton" in unit_text or "tonne" in unit_text:
        return amount * 1000.0
    if "kg" in unit_text or unit_text.strip() == "":
        return amount
    return None


def _convert_volume_to_m3(amount: float, unit_text: str) -> float | None:
    if "m3" in unit_text or "m^3" in unit_text:
        return amount
    if "ml" in unit_text:
        return amount / 1_000_000.0
    if " l" in f" {unit_text}" or unit_text.endswith("l"):
        return amount / 1000.0
    return None


PHYSICAL_BOUNDARIES: dict[str, tuple[float, float]] = {
    "caso4_fraction": (0.0, 1.0),
    "p2o5_fraction": (0.0, 0.2),  # P2O5 is an impurity, rarely exceeds 20%
    "f_fraction": (0.0, 0.1),  # Fluorine rarely exceeds 10%
    "moisture_fraction": (0.0, 1.0),
    "process_yield_fraction": (0.0, 1.0),
    "ra226_bq_kg": (0.0, 2000.0),  # Radioactivity Bq/kg is usually < 2000
    "electricity_kwh_per_t": (0.0, 5000.0),  # Max 5000 kWh/t
    "steam_mj_per_t": (0.0, 20000.0),  # Max 20000 MJ/t
    "process_water_m3_per_t": (0.0, 500.0),  # Max 500 m3/t
    "h2so4_kg_per_t": (0.0, 2000.0),
    "sodium_hydroxide_kg_per_t": (0.0, 2000.0),
    "ammonia_kg_per_t": (0.0, 2000.0),
    "co2_input_kg_per_t": (0.0, 3000.0),
    "extractant_l_per_t": (0.0, 1000.0),
    "binder_kg_per_t": (0.0, 2000.0),
    "leachate_m3_per_t": (0.0, 500.0),
    "residue_treatment_cost_usd_t": (0.0, 1000.0),
    "ree_market_price_usd_kg": (0.0, 10000.0),
    "trl": (1.0, 9.0),
}


def validate_value(key: str, value: float) -> bool:
    """Validate if value is within physical boundary limits."""
    if key in PHYSICAL_BOUNDARIES:
        low, high = PHYSICAL_BOUNDARIES[key]
        return low <= value <= high
    return True


def _append_metric(metrics: dict[str, list[float]], key: str, value: float | None) -> None:
    if value is None:
        return
    if not math.isfinite(value):
        return
    if not validate_value(key, value):
        print(
            f"    [Anomaly Rejected] {key} value {value} is outside physical boundaries {PHYSICAL_BOUNDARIES[key]}"
        )
        return
    metrics[key].append(float(value))


def _append_fraction_metric(metrics: dict[str, list[float]], key: str, raw_value: Any) -> None:
    number = _extract_first_number(raw_value)
    if number is None:
        return
    normalized = _normalize_fraction(number)
    _append_metric(metrics, key, normalized)


def _extract_composition_metrics(data: dict[str, Any], metrics: dict[str, list[float]]) -> None:
    composition = data.get("composition")
    if not isinstance(composition, dict):
        return
    _append_fraction_metric(metrics, "caso4_fraction", _get_field(composition, "CaSO4", "caso4"))
    _append_fraction_metric(metrics, "p2o5_fraction", _get_field(composition, "P2O5", "p2o5"))
    _append_fraction_metric(metrics, "f_fraction", _get_field(composition, "F", "f"))
    _append_fraction_metric(metrics, "moisture_fraction", _get_field(composition, "moisture"))
    ra226_value = _extract_first_number(_get_field(composition, "ra226", "Ra226"))
    _append_metric(metrics, "ra226_bq_kg", ra226_value)


def _extract_technology_metrics(data: dict[str, Any], metrics: dict[str, list[float]]) -> None:
    technology = data.get("technology")
    if not isinstance(technology, dict):
        return
    trl = _extract_first_number(_get_field(technology, "trl"))
    _append_metric(metrics, "trl", trl)


def _extract_energy_metrics(
    lci: dict[str, Any],
    metrics: dict[str, list[float]],
    fu_scale: float,
) -> None:
    energy = _get_field(lci, "energy_consumption")
    if not isinstance(energy, dict):
        return

    electricity_values: list[float] = []
    steam_values: list[float] = []
    unknown_energy_values: list[float] = []

    for source_name, raw_value in energy.items():
        amount = _extract_first_number(raw_value)
        if amount is None:
            continue
        source_text = _lower_text(source_name)
        unit_text = f"{source_text} {_lower_text(raw_value)}"

        kwh_amount = _convert_energy_to_kwh(amount, unit_text)
        mj_amount = _convert_energy_to_mj(amount, unit_text)
        kwh_per_t = _scale_to_per_tonne(kwh_amount, unit_text, fu_scale)
        mj_per_t = _scale_to_per_tonne(mj_amount, unit_text, fu_scale)

        if any(token in source_text for token in ("electric", "power", "grid")):
            electricity_values.append(kwh_per_t)
        elif any(token in source_text for token in ("steam", "heat", "thermal")):
            steam_values.append(mj_per_t)
        else:
            unknown_energy_values.append(kwh_per_t)

    for value in electricity_values:
        _append_metric(metrics, "electricity_kwh_per_t", value)
    for value in steam_values:
        _append_metric(metrics, "steam_mj_per_t", value)

    if not electricity_values and len(unknown_energy_values) == 1:
        _append_metric(metrics, "electricity_kwh_per_t", unknown_energy_values[0])


def _extract_chemical_metrics(
    lci: dict[str, Any],
    metrics: dict[str, list[float]],
    fu_scale: float,
) -> None:
    chemicals = _get_field(lci, "chemical_consumption")
    if not isinstance(chemicals, list):
        return

    for item in chemicals:
        if not isinstance(item, dict):
            continue
        name = _lower_text(_get_field(item, "name"))
        amount_raw = _get_field(item, "amount", "value")
        unit_raw = _get_field(item, "unit")
        amount = _extract_first_number(amount_raw)
        if amount is None:
            continue
        unit_text = f"{_lower_text(unit_raw)} {_lower_text(amount_raw)}"

        mass_kg = _convert_mass_to_kg(amount, unit_text)
        volume_m3 = _convert_volume_to_m3(amount, unit_text)

        mass_per_t = (
            _scale_to_per_tonne(mass_kg, unit_text, fu_scale) if mass_kg is not None else None
        )
        volume_per_t = (
            _scale_to_per_tonne(volume_m3, unit_text, fu_scale) if volume_m3 is not None else None
        )

        if any(token in name for token in ("h2so4", "sulfuric acid")):
            _append_metric(metrics, "h2so4_kg_per_t", mass_per_t)
        if any(token in name for token in ("naoh", "sodium hydroxide")):
            _append_metric(metrics, "sodium_hydroxide_kg_per_t", mass_per_t)
        if "ammonia" in name:
            _append_metric(metrics, "ammonia_kg_per_t", mass_per_t)
        if "co2" in name:
            _append_metric(metrics, "co2_input_kg_per_t", mass_per_t)
        if any(token in name for token in ("extractant", "solvent")):
            if volume_per_t is not None:
                _append_metric(metrics, "extractant_l_per_t", volume_per_t * 1000.0)
        if any(token in name for token in ("binder", "cement", "lime")):
            _append_metric(metrics, "binder_kg_per_t", mass_per_t)


def _extract_water_and_yield_metrics(
    lci: dict[str, Any],
    metrics: dict[str, list[float]],
    fu_scale: float,
) -> None:
    water = _get_field(lci, "water_consumption")
    water_amount = _extract_first_number(water)
    if water_amount is not None:
        water_text = _lower_text(water)
        water_m3 = _convert_volume_to_m3(water_amount, water_text)
        if water_m3 is None:
            water_kg = _convert_mass_to_kg(water_amount, water_text)
            water_m3 = water_kg / 1000.0 if water_kg is not None else None
        if water_m3 is not None:
            water_per_t = _scale_to_per_tonne(water_m3, water_text, fu_scale)
            _append_metric(metrics, "process_water_m3_per_t", water_per_t)

    yield_raw = _get_field(lci, "yield")
    _append_fraction_metric(metrics, "process_yield_fraction", yield_raw)

    emissions_water = _get_field(lci, "emissions_water")
    if isinstance(emissions_water, list):
        for item in emissions_water:
            if not isinstance(item, dict):
                continue
            substance = _lower_text(_get_field(item, "substance"))
            amount = _extract_first_number(_get_field(item, "amount"))
            unit_text = _lower_text(_get_field(item, "unit"))
            if amount is None:
                continue
            if "wastewater" in substance or "effluent" in substance:
                volume_m3 = _convert_volume_to_m3(amount, unit_text)
                if volume_m3 is not None:
                    _append_metric(
                        metrics,
                        "leachate_m3_per_t",
                        _scale_to_per_tonne(volume_m3, unit_text, fu_scale),
                    )


def _extract_cost_metrics(data: dict[str, Any], metrics: dict[str, list[float]]) -> None:
    cost = data.get("cost")
    if not isinstance(cost, dict):
        return

    opex = _get_field(cost, "opex")
    opex_value = _extract_first_number(opex)
    if opex_value is not None:
        opex_text = _lower_text(opex)
        if "waste" in opex_text or "residue" in opex_text:
            if "/kg" in opex_text:
                _append_metric(metrics, "residue_treatment_cost_usd_t", opex_value * 1000.0)
            elif "/t" in opex_text or "per tonne" in opex_text:
                _append_metric(metrics, "residue_treatment_cost_usd_t", opex_value)

    revenue = _get_field(cost, "revenue")
    revenue_value = _extract_first_number(revenue)
    if revenue_value is not None:
        revenue_text = _lower_text(revenue)
        if "/kg" in revenue_text:
            _append_metric(metrics, "ree_market_price_usd_kg", revenue_value)
        elif "/t" in revenue_text or "per tonne" in revenue_text:
            _append_metric(metrics, "ree_market_price_usd_kg", revenue_value / 1000.0)


def extract_metrics_from_record(data: dict[str, Any]) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = defaultdict(list)
    _extract_composition_metrics(data, metrics)
    _extract_technology_metrics(data, metrics)

    lci = data.get("lci")
    if isinstance(lci, dict):
        fu_scale = _functional_unit_scale_to_tonne(_get_field(lci, "functional_unit"))
        _extract_energy_metrics(lci, metrics, fu_scale)
        _extract_chemical_metrics(lci, metrics, fu_scale)
        _extract_water_and_yield_metrics(lci, metrics, fu_scale)

    _extract_cost_metrics(data, metrics)
    return metrics


def _percentile(sorted_values: list[float], percentile_rank: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = max(0.0, min(1.0, percentile_rank)) * (len(sorted_values) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _build_distribution(values: list[float]) -> dict[str, float | str]:
    sorted_values = sorted(values)
    lo = sorted_values[0]
    hi = sorted_values[-1]
    med = median(sorted_values)
    if math.isclose(lo, hi, rel_tol=1e-12, abs_tol=1e-12):
        return {"type": "fixed", "value": med}
    return {"type": "triangular", "min": lo, "mode": med, "max": hi}


def _fallback_distribution(default_value: float) -> dict[str, float | str]:
    if default_value <= 0:
        return {"type": "fixed", "value": default_value}
    return {
        "type": "triangular",
        "min": default_value * 0.8,
        "mode": default_value,
        "max": default_value * 1.2,
    }


def _stats(values: list[float], unit: str | None = None) -> dict[str, Any]:
    sorted_values = sorted(values)
    p10 = _percentile(sorted_values, 0.10)
    p90 = _percentile(sorted_values, 0.90)
    med = median(sorted_values)
    result: dict[str, Any] = {
        "sample_count": len(sorted_values),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": mean(sorted_values),
        "median": med,
        "p10": p10,
        "p90": p90,
        "distribution": _build_distribution(sorted_values),
    }
    if unit:
        result["unit"] = unit
    return result


def _safe_json_load(path: Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            loaded = json.load(file_obj)
        if isinstance(loaded, dict):
            return loaded
    except (OSError, json.JSONDecodeError):
        return {}
    return {}


def _collect_global_samples(
    json_files: list[Path],
) -> tuple[dict[str, list[float]], dict[str, set[str]], int]:
    global_samples: dict[str, list[float]] = defaultdict(list)
    sources: dict[str, set[str]] = defaultdict(set)
    valid_docs = 0
    for json_path in json_files:
        record = _safe_json_load(json_path)
        if not record:
            continue
        valid_docs += 1
        metric_values = extract_metrics_from_record(record)
        for key, values in metric_values.items():
            for value in values:
                if math.isfinite(value):
                    global_samples[key].append(float(value))
                    sources[key].add(json_path.name)
    return global_samples, sources, valid_docs


def _build_global_parameter_report(
    global_samples: dict[str, list[float]],
    sources: dict[str, set[str]],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for parameter, values in sorted(global_samples.items()):
        if not values:
            continue
        unit = PARAMETER_UNITS.get(parameter)
        report[parameter] = _stats(values, unit=unit)
        report[parameter]["source_files"] = sorted(sources.get(parameter, set()))
    return report


def _merge_with_mapping_rules(
    source_samples: dict[str, list[float]],
    rules: list[tuple[str, Callable[[float], float]]],
) -> tuple[list[float], list[str]]:
    values: list[float] = []
    source_names: list[str] = []
    for source_param, transform in rules:
        source_values = source_samples.get(source_param, [])
        if not source_values:
            continue
        transformed_values = [transform(value) for value in source_values]
        values.extend(v for v in transformed_values if math.isfinite(v))
        source_names.append(source_param)
    return values, sorted(set(source_names))


def _build_pathway_profiles(global_samples: dict[str, list[float]]) -> dict[str, Any]:
    pathways_output: dict[str, Any] = {}

    # Try importing and initializing GapFiller
    try:
        from pgloop.knowledge.gap_filler import SKLEARN_AVAILABLE, GapFiller

        gap_filler = GapFiller() if SKLEARN_AVAILABLE else None
    except Exception:
        gap_filler = None

    for pathway_code, pathway_cls in PATHWAYS.items():
        pathway = pathway_cls()
        defaults = pathway.parameters
        mapping_rules = PATHWAY_MAPPING_RULES.get(pathway_code, {})

        parameters: dict[str, Any] = {}
        distributions: dict[str, Any] = {}
        inferred_count = 0
        missing_parameters: list[str] = []
        known_parameters: dict[str, float] = {}

        # First pass: extract parameters matching rules
        for parameter_name, default_value in defaults.items():
            rules = mapping_rules.get(parameter_name, [(parameter_name, _identity)])
            inferred_values, source_names = _merge_with_mapping_rules(global_samples, rules)
            unit = PARAMETER_UNITS.get(parameter_name)

            if inferred_values:
                inferred_count += 1
                stat = _stats(inferred_values, unit=unit)
                parameters[parameter_name] = {
                    "value": stat["median"],
                    "range": [stat["min"], stat["max"]],
                    "sample_count": stat["sample_count"],
                    "default_value": default_value,
                    "fallback_used": False,
                    "source_parameters": source_names,
                }
                if unit:
                    parameters[parameter_name]["unit"] = unit
                distributions[parameter_name] = stat["distribution"]
                known_parameters[parameter_name] = stat["median"]
            else:
                missing_parameters.append(parameter_name)

        # Second pass: fill missing parameters using GapFiller or default fallback
        for parameter_name in missing_parameters:
            default_value = defaults[parameter_name]
            unit = PARAMETER_UNITS.get(parameter_name)
            pred_used = False

            if gap_filler is not None and known_parameters:
                try:
                    pred_res = gap_filler.predict_by_similarity(known_parameters, parameter_name)
                    if pred_res and not math.isnan(pred_res.predicted_value):
                        low = (
                            pred_res.uncertainty_low
                            if math.isfinite(pred_res.uncertainty_low)
                            else pred_res.predicted_value * 0.8
                        )
                        high = (
                            pred_res.uncertainty_high
                            if math.isfinite(pred_res.uncertainty_high)
                            else pred_res.predicted_value * 1.2
                        )
                        parameters[parameter_name] = {
                            "value": pred_res.predicted_value,
                            "range": [low, high],
                            "sample_count": 0,
                            "default_value": default_value,
                            "fallback_used": True,
                            "gap_filled": True,
                            "gap_fill_method": pred_res.method,
                            "source_parameters": [],
                        }
                        if unit:
                            parameters[parameter_name]["unit"] = unit
                        distributions[parameter_name] = {
                            "type": "triangular",
                            "min": low,
                            "mode": pred_res.predicted_value,
                            "max": high,
                        }
                        pred_used = True
                except Exception:
                    pass

            if not pred_used:
                parameters[parameter_name] = {
                    "value": default_value,
                    "range": [default_value, default_value],
                    "sample_count": 0,
                    "default_value": default_value,
                    "fallback_used": True,
                    "source_parameters": [],
                }
                if unit:
                    parameters[parameter_name]["unit"] = unit
                distributions[parameter_name] = _fallback_distribution(default_value)

        total_parameters = len(defaults) if defaults else 1
        coverage = inferred_count / total_parameters
        pathways_output[pathway_code] = {
            "coverage": coverage,
            "inferred_parameters": inferred_count,
            "total_parameters": len(defaults),
            "missing_parameters": missing_parameters,
            "parameters": parameters,
            "parameter_distributions": distributions,
        }
    return pathways_output


def build_parameter_ranges_from_extracted(
    extracted_dir: Path,
    output_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """
    Build parameter ranges from extracted JSON files and save report files.

    Returns:
        A summary dict with generated file paths and high-level statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(extracted_dir.glob("*_extracted.json"))
    if limit is not None:
        json_files = json_files[:limit]

    global_samples, sources, valid_docs = _collect_global_samples(json_files)
    global_report = _build_global_parameter_report(global_samples, sources)
    pathway_profiles = _build_pathway_profiles(global_samples)

    generated_at = datetime.now().isoformat(timespec="seconds")
    range_report_path = output_dir / "parameter_ranges.json"
    pathway_profile_path = output_dir / "pathway_parameter_profiles.json"
    pathway_distribution_path = output_dir / "pathway_parameter_distributions.json"

    range_report_payload = {
        "generated_at": generated_at,
        "source_directory": str(extracted_dir),
        "source_file_count": len(json_files),
        "valid_document_count": valid_docs,
        "notes": (
            "Values are normalized to per-tonne basis when unit text allows detection. "
            "Fallback values remain pathway defaults."
        ),
        "global_parameters": global_report,
    }
    with open(range_report_path, "w", encoding="utf-8") as file_obj:
        json.dump(range_report_payload, file_obj, indent=2, ensure_ascii=False)

    pathway_profile_payload = {
        "generated_at": generated_at,
        "source_directory": str(extracted_dir),
        "source_file_count": len(json_files),
        "pathways": pathway_profiles,
    }
    with open(pathway_profile_path, "w", encoding="utf-8") as file_obj:
        json.dump(pathway_profile_payload, file_obj, indent=2, ensure_ascii=False)

    distributions_only = {
        pathway_code: details["parameter_distributions"]
        for pathway_code, details in pathway_profiles.items()
    }
    with open(pathway_distribution_path, "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "generated_at": generated_at,
                "source_directory": str(extracted_dir),
                "source_file_count": len(json_files),
                "distributions": distributions_only,
            },
            file_obj,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "source_file_count": len(json_files),
        "valid_document_count": valid_docs,
        "global_parameter_count": len(global_report),
        "output_files": {
            "ranges": str(range_report_path),
            "profiles": str(pathway_profile_path),
            "distributions": str(pathway_distribution_path),
        },
    }
