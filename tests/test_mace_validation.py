"""
MACE Model Validation Test Suite

Validates:
1. MACE calculator initialization and presence.
2. Crystalline structure relaxation (BFGS / ExpCellFilter).
3. Equation of State (EOS) isotropic scaling and Birch-Murnaghan fitting.
4. Materials Project API retrieval and accuracy benchmarking (< 0.05 eV/atom).
"""

import os
import pytest
import numpy as np
from ase.build import bulk

# Skip entire test file if mace is not installed or importable
mace = pytest.importorskip("mace")

from pgloop.chemicals.mace_interface import get_mace_calculator, is_mace_available
from pgloop.chemicals.lattice_optimizer import optimize_structure, fit_eos
from pgloop.chemicals.eval_mace import evaluate_mace_on_mp, fetch_mp_data


def test_mace_calculator_loading():
    """Verify MACE calculator loads successfully and correctly sets up."""
    assert is_mace_available()
    calc = get_mace_calculator(model_size="small", device="cpu")
    assert calc is not None


def test_mace_static_and_relaxation():
    """Test static potential energy calculation and coordinate/cell relaxation using MACE."""
    # Create a small Copper FCC cell (a=3.6 Angstroms)
    atoms = bulk("Cu", "fcc", a=3.6)
    calc = get_mace_calculator(model_size="small", device="cpu")

    # 1. Single Point Static Energy
    atoms.calc = calc
    energy_before = atoms.get_potential_energy()
    assert energy_before is not None

    # 2. Relax structure (coordinates and cell parameters)
    opt_atoms, info = optimize_structure(
        atoms,
        calculator=calc,
        fmax=0.1,
        steps=10,
        constant_volume=False,
    )

    assert info["converged"] or info["steps"] > 0
    energy_after = opt_atoms.get_potential_energy()

    # Energy should decrease or stay the same after relaxation
    assert energy_after <= energy_before
    assert len(opt_atoms) == len(atoms)
    assert opt_atoms.get_volume() > 0


def test_mace_eos_fitting():
    """Test Equation of State (EOS) isotropic cell scaling and fitting."""
    atoms = bulk("Cu", "fcc", a=3.6)
    calc = get_mace_calculator(model_size="small", device="cpu")

    eos, results = fit_eos(
        atoms,
        calculator=calc,
        num_points=5,
        strain_range=0.03,
        eos_type="birchmurnaghan",
    )

    # Validate volume and energy arrays
    assert len(results["volumes_ang3"]) == 5
    assert len(results["energies_ev"]) == 5

    # Check fitted values
    assert results["v0_ang3"] > 0.0
    assert results["b0_gpa"] > 0.0
    assert results["eos_type"] == "birchmurnaghan"


@pytest.mark.skipif(
    not os.environ.get("MP_API_KEY"),
    reason="Materials Project API key (MP_API_KEY) not configured in the environment."
)
def test_mace_error_on_mp():
    """
    Evaluate MACE energy predictions on a real crystal structure from Materials Project
    and verify the mean absolute error is strictly less than 0.05 eV/atom.
    """
    # mp-4406 is CaSO4 (Anhydrite)
    material_ids = ["mp-4406"]
    
    # We run evaluation on cpu and small model size for testing speed
    eval_results = evaluate_mace_on_mp(
        material_ids=material_ids,
        model_size="small",
        device="cpu",
    )

    assert eval_results["success"] is True
    assert eval_results["mae"] < 0.05
    assert len(eval_results["results"]) == 1
    
    record = eval_results["results"][0]
    assert record["material_id"] == "mp-4406"
    assert record["error_ev_atom"] < 0.05
