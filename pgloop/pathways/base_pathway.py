"""
Base Pathway Module

Abstract base class for all treatment pathways.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import copy

from pgloop.lca.inventory import LifeCycleInventory


@dataclass
class PathwayConfig:
    """Configuration for a treatment pathway."""
    
    name: str
    code: str
    description: str = ""
    trl: int = 9
    country: str = "global"
    year: int = 2024
    annual_capacity_tonnes: float = 100000


class BasePathway(ABC):
    """
    Abstract base class for phosphogypsum treatment pathways.
    
    Each pathway must define:
    - LCI data (inputs, outputs, emissions)
    - CAPEX data (equipment, installation)
    - OPEX data (materials, utilities, labor)
    - Product data (outputs with prices)
    """
    
    def __init__(
        self,
        country: str = "global",
        year: int = 2024,
        capacity_tonnes: float = 100000
    ):
        """
        Initialize pathway.
        
        Args:
            country: Country context for costs
            year: Reference year
            capacity_tonnes: Annual capacity in tonnes
        """
        self.country = country
        self.year = year
        self.capacity = capacity_tonnes
        self.parameters = self._default_parameters()
        self.inventory = self._build_inventory()
    
    @property
    @abstractmethod
    def code(self) -> str:
        """Pathway code (e.g., PG-SD, PG-CM)."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Pathway name."""
        pass
    
    @property
    def trl(self) -> int:
        """Technology Readiness Level."""
        return 9
    
    @abstractmethod
    def _default_parameters(self) -> Dict[str, float]:
        """Return default parameters for the pathway."""
        pass
    
    @abstractmethod
    def _build_inventory(self) -> LifeCycleInventory:
        """Build the life cycle inventory."""
        pass
    
    @abstractmethod
    def get_capex_data(self) -> Dict:
        """Return CAPEX data for TEA."""
        pass
    
    @abstractmethod
    def get_opex_data(self) -> Dict:
        """Return OPEX data for TEA."""
        pass
    
    @abstractmethod
    def get_products(self) -> List[Dict]:
        """Return product outputs with prices."""
        pass
    
    def get_scaled_inventory(
        self,
        functional_unit_kg: float,
        parameters: Dict = None
    ) -> Dict:
        """
        Get inventory scaled to functional unit.
        
        Args:
            functional_unit_kg: Functional unit in kg
            parameters: Optional parameter overrides
            
        Returns:
            Scaled inventory as dict
        """
        # Update parameters if provided
        if parameters:
            old_params = self.parameters.copy()
            self.parameters.update(parameters)
            self.inventory = self._build_inventory()
            self.parameters = old_params
        
        # Scale inventory
        scaled = self.inventory.scale_to(functional_unit_kg)
        return scaled.to_dict()
    
    def get_emissions(self) -> Dict:
        """Get emissions for external cost calculation."""
        inv = self.inventory.to_dict()
        return {
            "emissions_air": inv.get("emissions_air", []),
            "emissions_water": inv.get("emissions_water", []),
            "emissions_soil": inv.get("emissions_soil", []),
            "avoided_products": inv.get("avoided_products", [])
        }
    
    def get_annual_throughput(self) -> float:
        """Get annual throughput in tonnes."""
        return self.capacity
    
    def get_parameter_distributions(self) -> Dict[str, Dict]:
        """
        Get parameter distributions for uncertainty analysis.
        
        Returns:
            Dict mapping parameter to distribution spec
        """
        distributions = {}
        for param, value in self.parameters.items():
            # Default: triangular distribution ±20%
            distributions[param] = {
                "type": "triangular",
                "min": value * 0.8,
                "mode": value,
                "max": value * 1.2
            }
        return distributions
    
    def get_cost_distributions(self) -> Dict[str, Dict]:
        """Get cost parameter distributions for TEA uncertainty."""
        return self.get_parameter_distributions()
    
    def copy_with_modified_parameter(
        self,
        parameter: str,
        factor: float
    ) -> "BasePathway":
        """Create a copy with a modified parameter."""
        new_pathway = copy.deepcopy(self)
        if parameter in new_pathway.parameters:
            new_pathway.parameters[parameter] *= factor
            new_pathway.inventory = new_pathway._build_inventory()
        return new_pathway
    
    def copy_with_parameters(self, parameters: Dict) -> "BasePathway":
        """Create a copy with modified parameters."""
        new_pathway = copy.deepcopy(self)
        new_pathway.parameters.update(parameters)
        new_pathway.inventory = new_pathway._build_inventory()
        return new_pathway
    
    def to_dict(self) -> Dict:
        """Export pathway configuration as dict."""
        return {
            "code": self.code,
            "name": self.name,
            "trl": self.trl,
            "country": self.country,
            "year": self.year,
            "capacity_tonnes": self.capacity,
            "parameters": self.parameters,
            "inventory": self.inventory.to_dict()
        }
