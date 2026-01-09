"""
Property Predictor Module

Multi-backend property prediction with MACE force field integration.
Fallback hierarchy: Database -> MACE -> Group Contribution -> Default
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import warnings


@dataclass
class PropertyPrediction:
    """Result of a property prediction."""
    
    value: float
    unit: str
    source: str  # "database", "mace", "group_contribution", "default"
    uncertainty: Optional[float] = None
    temperature_k: float = 298.15


# Property units
PROPERTY_UNITS = {
    "density": "kg/m3",
    "heat_capacity": "J/(mol·K)",
    "viscosity": "Pa·s",
    "boiling_point": "K",
    "melting_point": "K",
    "vapor_pressure": "Pa",
    "formation_energy": "eV",
}


class MACEPredictor:
    """
    MACE Universal Force Field Predictor.
    
    Uses MACE-MP for molecular property prediction from SMILES.
    """
    
    def __init__(self, model: str = "medium", device: str = "cpu"):
        self.model_name = model
        self.device = device
        self._calculator = None
        self._available = None
    
    @property
    def is_available(self) -> bool:
        """Check if MACE is installed and usable."""
        if self._available is None:
            try:
                from mace.calculators import mace_mp
                self._available = True
            except ImportError:
                self._available = False
        return self._available
    
    def _get_calculator(self):
        """Lazy load MACE calculator."""
        if self._calculator is None and self.is_available:
            from mace.calculators import mace_mp
            self._calculator = mace_mp(model=self.model_name, device=self.device)
        return self._calculator
    
    def smiles_to_atoms(self, smiles: str):
        """Convert SMILES to ASE Atoms object."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from ase import Atoms
            import numpy as np
            
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Add hydrogens and generate 3D
            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
                return None
            AllChem.MMFFOptimizeMolecule(mol)
            
            # Extract positions and elements
            conf = mol.GetConformer()
            positions = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            
            return Atoms(symbols=symbols, positions=positions)
            
        except ImportError:
            warnings.warn("RDKit not installed. SMILES conversion unavailable.")
            return None
    
    def predict_energy(self, smiles: str) -> Optional[float]:
        """Predict formation energy using MACE."""
        if not self.is_available:
            return None
        
        atoms = self.smiles_to_atoms(smiles)
        if atoms is None:
            return None
        
        try:
            atoms.calc = self._get_calculator()
            return atoms.get_potential_energy()
        except Exception as e:
            warnings.warn(f"MACE energy prediction failed: {e}")
            return None
    
    def predict_property(
        self,
        smiles: str,
        property_type: str,
        temperature_k: float = 298.15
    ) -> Optional[float]:
        """
        Predict molecular property using MACE.
        
        Supported properties:
        - formation_energy: Direct from MACE
        - density: Estimated from molecular volume
        - heat_capacity: From vibrational analysis (simplified)
        """
        if not self.is_available:
            return None
        
        atoms = self.smiles_to_atoms(smiles)
        if atoms is None:
            return None
        
        try:
            atoms.calc = self._get_calculator()
            
            if property_type == "formation_energy":
                return atoms.get_potential_energy()
            
            elif property_type == "density":
                # Estimate from molecular volume
                from ase.geometry import get_distances
                positions = atoms.get_positions()
                if len(positions) < 2:
                    return 1000.0
                
                # Approximate molecular volume from bounding box
                bbox = positions.max(axis=0) - positions.min(axis=0)
                volume_A3 = max(bbox[0], 3) * max(bbox[1], 3) * max(bbox[2], 3)
                volume_m3 = volume_A3 * 1e-30
                
                # Mass from atom masses
                mass_kg = sum(atoms.get_masses()) * 1.66054e-27
                
                return mass_kg / volume_m3 if volume_m3 > 0 else 1000.0
            
            elif property_type == "heat_capacity":
                # Simplified: 3R per atom (Dulong-Petit approximation)
                R = 8.314  # J/(mol·K)
                n_atoms = len(atoms)
                return 3 * R * n_atoms
            
            else:
                return None
                
        except Exception as e:
            warnings.warn(f"MACE property prediction failed: {e}")
            return None


class GroupContributionPredictor:
    """
    Group Contribution Method for property estimation.
    
    Uses Joback-Reid method for thermodynamic properties.
    """
    
    # Joback group contributions for boiling point
    JOBACK_GROUPS = {
        "-CH3": {"Tb": 23.58, "Tc": 0.0141, "Cp_a": 19.5, "Cp_b": -0.00808},
        "-CH2-": {"Tb": 22.88, "Tc": 0.0189, "Cp_a": -0.909, "Cp_b": 0.0950},
        ">CH-": {"Tb": 21.74, "Tc": 0.0164, "Cp_a": -23.0, "Cp_b": 0.204},
        "-OH": {"Tb": 92.88, "Tc": 0.0741, "Cp_a": 25.7, "Cp_b": -0.0691},
        "-COOH": {"Tb": 169.09, "Tc": 0.0791, "Cp_a": 24.1, "Cp_b": 0.0427},
        "-NH2": {"Tb": 73.23, "Tc": 0.0243, "Cp_a": 26.9, "Cp_b": -0.0412},
    }
    
    def predict_property(
        self,
        smiles: str,
        property_type: str,
        temperature_k: float = 298.15
    ) -> Optional[float]:
        """Predict property using group contribution method."""
        # Simplified implementation - would need full group detection
        # This is a placeholder for the full Joback-Reid method
        
        if property_type == "boiling_point":
            # Very rough estimate based on molecular size
            n_heavy = len([c for c in smiles if c.isupper()])
            return 273 + 30 * n_heavy  # Rough approximation
        
        elif property_type == "heat_capacity":
            # Rough estimate
            n_atoms = len([c for c in smiles if c.isalpha()])
            return 30 + 5 * n_atoms  # J/(mol·K)
        
        return None


class PropertyPredictor:
    """
    Unified property prediction engine.
    
    Combines multiple prediction backends with intelligent fallback.
    """
    
    def __init__(
        self,
        use_mace: bool = True,
        use_group_contribution: bool = True,
        cache_predictions: bool = True
    ):
        self.use_mace = use_mace
        self.use_gc = use_group_contribution
        
        self._mace = MACEPredictor() if use_mace else None
        self._gc = GroupContributionPredictor() if use_group_contribution else None
        self._cache: Dict[str, PropertyPrediction] = {} if cache_predictions else None
    
    def get_property(
        self,
        smiles: str,
        property_name: str,
        temperature_k: float = 298.15
    ) -> PropertyPrediction:
        """
        Get property value using best available method.
        
        Fallback order: Cache -> MACE -> Group Contribution -> Default
        
        Args:
            smiles: SMILES representation
            property_name: Property to predict
            temperature_k: Temperature
            
        Returns:
            PropertyPrediction with value and source
        """
        cache_key = f"{smiles}_{property_name}_{temperature_k}"
        
        # 1. Check cache
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]
        
        unit = PROPERTY_UNITS.get(property_name, "")
        
        # 2. Try MACE
        if self._mace and self._mace.is_available:
            value = self._mace.predict_property(smiles, property_name, temperature_k)
            if value is not None:
                result = PropertyPrediction(
                    value=value,
                    unit=unit,
                    source="mace",
                    uncertainty=0.1,
                    temperature_k=temperature_k
                )
                if self._cache is not None:
                    self._cache[cache_key] = result
                return result
        
        # 3. Try Group Contribution
        if self._gc:
            value = self._gc.predict_property(smiles, property_name, temperature_k)
            if value is not None:
                result = PropertyPrediction(
                    value=value,
                    unit=unit,
                    source="group_contribution",
                    uncertainty=0.2,
                    temperature_k=temperature_k
                )
                if self._cache is not None:
                    self._cache[cache_key] = result
                return result
        
        # 4. Return default
        defaults = {
            "density": 1000.0,
            "heat_capacity": 75.0,
            "viscosity": 0.001,
            "boiling_point": 373.15,
            "melting_point": 273.15,
        }
        
        return PropertyPrediction(
            value=defaults.get(property_name, 0.0),
            unit=unit,
            source="default",
            uncertainty=0.5,
            temperature_k=temperature_k
        )
    
    def clear_cache(self) -> None:
        """Clear prediction cache."""
        if self._cache is not None:
            self._cache.clear()
