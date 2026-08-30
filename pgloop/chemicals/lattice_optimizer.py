"""
Lattice Optimizer & Equation of State (EOS) Engine

Uses ASE's BFGS/FIRE optimizers and ExpCellFilter to relax crystalline
structures (positions, cell shapes, cell volumes), and provides Birch-Murnaghan
fitting of Equation of State (EOS) to determine bulk modulus properties.
"""

import numpy as np
from typing import Dict, Any, Tuple, List

try:
    from ase import Atoms
    from ase.optimize import BFGS
    from ase.filters import ExpCellFilter
    from ase.eos import EquationOfState
    ASE_AVAILABLE = True
except ImportError:
    Atoms = Any
    BFGS = None
    ExpCellFilter = None
    EquationOfState = None
    ASE_AVAILABLE = False


def optimize_structure(
    atoms: Atoms,
    calculator: Any,
    optimizer_class: Any = BFGS,
    fmax: float = 0.05,
    steps: int = 100,
    constant_volume: bool = False,
    hydrostatic_strain: bool = False,
    logfile: str = None,
) -> Tuple[Atoms, Dict[str, Any]]:
    """
    Optimize geometry (atomic coordinates, cell shape, and/or volume) of a crystal.

    Args:
        atoms (Atoms): Input ASE Atoms object.
        calculator (Any): ASE calculator (e.g., MACE).
        optimizer_class (Any): ASE optimizer class (e.g., BFGS, FIRE).
        fmax (float): Maximum residual force threshold for convergence (eV/Angstrom).
        steps (int): Maximum optimization steps.
        constant_volume (bool): If True, only optimize atomic coordinates.
                                 If False, optimize coordinates and cell parameters.
        hydrostatic_strain (bool): If True, constrain cell optimization to isotropic scaling.
        logfile (str, optional): File path to write optimization logs to.

    Returns:
        Tuple[Atoms, dict]: Optimized structure and metadata dict.
    """
    atoms_copy = atoms.copy()
    atoms_copy.calc = calculator

    if constant_volume:
        opt_target = atoms_copy
    else:
        # ExpCellFilter allows simultaneous coordinate and cell relaxation
        opt_target = ExpCellFilter(atoms_copy, hydrostatic_strain=hydrostatic_strain)

    opt = optimizer_class(opt_target, logfile=logfile)
    converged = opt.run(fmax=fmax, steps=steps)

    # Clean up calculator reference on the optimizer target
    # Calculate final properties
    final_energy = atoms_copy.get_potential_energy()
    final_forces = atoms_copy.get_forces()
    final_stress = atoms_copy.get_stress() if not constant_volume else None

    result_metadata = {
        "converged": converged,
        "steps": opt.nsteps,
        "energy_ev": final_energy,
        "energy_per_atom_ev": final_energy / len(atoms_copy),
        "forces": final_forces,
        "stress": final_stress,
        "volume_ang3": atoms_copy.get_volume(),
    }

    return atoms_copy, result_metadata


def fit_eos(
    atoms: Atoms,
    calculator: Any,
    num_points: int = 7,
    strain_range: float = 0.05,
    eos_type: str = "birchmurnaghan",
) -> Tuple[EquationOfState, Dict[str, Any]]:
    """
    Perform isotropic cell scaling, compute energies, and fit an Equation of State (EOS).

    Args:
        atoms (Atoms): Base structure (usually relaxed).
        calculator (Any): ASE calculator.
        num_points (int): Number of volume points to compute (must be >= 4).
        strain_range (float): Maximum scaling factor deviation (e.g., 0.05 is +/-5% strain).
        eos_type (str): Type of EOS to fit ('birchmurnaghan', 'murnaghan', etc.).

    Returns:
        Tuple[EquationOfState, dict]: Fitted ASE EquationOfState object and results dict.
    """
    if num_points < 4:
        raise ValueError("Equation of State fitting requires at least 4 data points.")

    atoms_copy = atoms.copy()
    atoms_copy.calc = calculator

    original_cell = atoms_copy.get_cell()
    scale_factors = np.linspace(1.0 - strain_range, 1.0 + strain_range, num_points)

    volumes = []
    energies = []

    for scale in scale_factors:
        atoms_copy.set_cell(original_cell * scale, scale_atoms=True)
        energy = atoms_copy.get_potential_energy()
        volume = atoms_copy.get_volume()
        volumes.append(volume)
        energies.append(energy)

    # Restore original cell
    atoms_copy.set_cell(original_cell, scale_atoms=True)

    # Fit Birch-Murnaghan EOS
    eos = EquationOfState(volumes, energies, eos_type)
    v0, e0, b0 = eos.fit()

    # Convert Bulk Modulus from eV/Angstrom^3 to GPa
    # 1 eV/Ang3 = 160.21766208 GPa
    b0_gpa = b0 * 160.21766208

    results = {
        "volumes_ang3": volumes,
        "energies_ev": energies,
        "v0_ang3": v0,
        "e0_ev": e0,
        "b0_ev_ang3": b0,
        "b0_gpa": b0_gpa,
        "eos_type": eos_type,
    }

    return eos, results
