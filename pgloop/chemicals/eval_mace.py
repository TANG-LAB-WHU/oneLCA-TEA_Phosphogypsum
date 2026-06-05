"""
Materials Project & MACE Potential Evaluation Engine

Fetches real DFT crystal structures and reference energy properties online
using the Materials Project API (mp-api), computes static potential energies
via MACE-MP, and evaluates performance against the KPI error threshold.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv

# Try importing mp-api & pymatgen
try:
    from mp_api.client import MPRester
    MP_API_AVAILABLE = True
except ImportError:
    MPRester = None
    MP_API_AVAILABLE = False

try:
    from pymatgen.io.ase import AseAtomsAdaptor
    PYMATGEN_AVAILABLE = True
except ImportError:
    AseAtomsAdaptor = None
    PYMATGEN_AVAILABLE = False

from pgloop.chemicals.mace_interface import get_mace_calculator, is_mace_available


def fetch_mp_data(material_ids: List[str], api_key: str = None) -> List[Dict[str, Any]]:
    """
    Fetch crystal structure and DFT energy benchmarks from the Materials Project API.

    Args:
        material_ids (List[str]): List of MP material IDs (e.g. ['mp-4406', 'mp-2741']).
        api_key (str, optional): Materials Project API key. Defaults to MP_API_KEY env var.

    Returns:
        List[Dict[str, Any]]: List of dictionary structures containing the fetched benchmarks.
    """
    load_dotenv()
    key = api_key or os.environ.get("MP_API_KEY")

    if not key:
        raise ValueError(
            "Materials Project API key is missing. Please set the 'MP_API_KEY' "
            "environment variable in your '.env' file or current terminal session."
        )

    if not MP_API_AVAILABLE:
        raise ImportError(
            "The 'mp-api' package is not available. Install it using 'pip install mp-api'."
        )

    if not PYMATGEN_AVAILABLE:
        raise ImportError(
            "The 'pymatgen' package is not available. Install it using 'pip install pymatgen'."
        )

    results = []
    print(f"[INFO] Connecting to Materials Project online database to fetch {len(material_ids)} structures...")

    with MPRester(key) as mpr:
        # Search for structures and energies
        docs = mpr.summary.search(
            material_ids=material_ids,
            fields=["material_id", "structure", "energy", "energy_per_atom", "formula_pretty"]
        )

        for doc in docs:
            mat_id = str(getattr(doc, "material_id"))
            struct = getattr(doc, "structure")
            dft_energy = getattr(doc, "energy")
            dft_energy_per_atom = getattr(doc, "energy_per_atom")
            formula = getattr(doc, "formula_pretty")

            results.append({
                "material_id": mat_id,
                "structure": struct,
                "dft_energy_ev": dft_energy,
                "dft_energy_per_atom_ev": dft_energy_per_atom,
                "formula": formula,
            })

    # Validate that we retrieved what was requested
    retrieved_ids = {item["material_id"] for item in results}
    missing_ids = set(material_ids) - retrieved_ids
    if missing_ids:
        print(f"[WARNING] Could not retrieve the following MP IDs: {missing_ids}")

    return results


def evaluate_mace_on_mp(
    material_ids: List[str],
    model_size: str = "medium",
    device: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    """
    Download structures from MP, evaluate potential energy using MACE, and compute error metrics.

    Args:
        material_ids (List[str]): List of MP material IDs to evaluate.
        model_size (str): Size of the MACE model to load ('small', 'medium', 'large').
        device (str, optional): Hardware device ('cpu', 'cuda').
        api_key (str, optional): MP API key.

    Returns:
        Dict[str, Any]: Dictionary of evaluation results, metrics, and KPI checks.
    """
    if not is_mace_available():
        raise ImportError(
            "MACE calculator is not available. Install dependencies using 'pip install mace-torch'."
        )

    # Fetch reference DFT data
    mp_data = fetch_mp_data(material_ids, api_key=api_key)
    if not mp_data:
        raise ValueError("No valid benchmark data could be fetched from Materials Project.")

    # Initialize MACE calculator
    calc = get_mace_calculator(model_size=model_size, device=device)

    errors = []
    evaluation_records = []

    print("\n[INFO] Starting MACE potential prediction validation...")
    print("-" * 80)
    print(f"{'Formula':<15} | {'MP ID':<10} | {'DFT (eV/atom)':<15} | {'MACE (eV/atom)':<15} | {'Error (eV/atom)':<15}")
    print("-" * 80)

    for item in mp_data:
        structure = item["structure"]
        dft_epa = item["dft_energy_per_atom_ev"]
        formula = item["formula"]
        mat_id = item["material_id"]

        # Convert to ASE Atoms and assign calculator
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = calc

        # Compute static energy
        mace_total_energy = atoms.get_potential_energy()
        num_atoms = len(atoms)
        mace_epa = mace_total_energy / num_atoms

        error_epa = abs(mace_epa - dft_epa)
        errors.append(error_epa)

        print(f"{formula:<15} | {mat_id:<10} | {dft_epa:15.6f} | {mace_epa:15.6f} | {error_epa:15.6f}")

        evaluation_records.append({
            "material_id": mat_id,
            "formula": formula,
            "num_atoms": num_atoms,
            "dft_energy_per_atom_ev": dft_epa,
            "mace_energy_per_atom_ev": mace_epa,
            "error_ev_atom": error_epa,
        })

    print("-" * 80)

    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    success = mae < 0.05

    print(f"Evaluation Metrics Summary:")
    print(f"  * Mean Absolute Error (MAE):  {mae:.6f} eV/atom")
    print(f"  * Root Mean Square Error (RMSE): {rmse:.6f} eV/atom")
    print(f"  * Success Criteria Check (< 0.05 eV/atom): {'PASS' if success else 'FAIL'}")
    print("-" * 80)

    return {
        "mae": mae,
        "rmse": rmse,
        "success": success,
        "results": evaluation_records,
    }


def main():
    # Primary Phosphogypsum phases and co-existing impurities:
    # 1. CaSO4 (Anhydrite): mp-4406
    # 2. CaSO4.2H2O (Gypsum): mp-23690
    # 3. CaF2 (Fluorite): mp-2741
    default_ids = ["mp-4406", "mp-23690", "mp-2741"]

    print("[INFO] Running default MACE online evaluation suite...")
    try:
        evaluate_mace_on_mp(default_ids, model_size="medium")
    except Exception as err:
        print(f"\n[ERROR] Evaluation execution failed: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

