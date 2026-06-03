from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel

class VPMSchema(BaseModel):
    """Base schema for inputs and outputs of a VPM."""
    pass

class ValidationReport(BaseModel):
    """Report generated after validating the VPM against benchmarks."""
    is_valid: bool
    metrics: Dict[str, float]
    details: str

class ValorizationPathwayModule(ABC):
    """
    Abstract interface for all Phosphogypsum treatment pathway PINN models.
    Enforces a strict modular contract for I/O and physics.
    """
    
    @property
    @abstractmethod
    def module_id(self) -> str:
        """Unique identifier for the module (e.g., 'VPM_carbothermic_reduction')."""
        pass
    
    @property  
    @abstractmethod
    def governing_equations(self) -> List[str]:
        """Returns the list of governing PDEs/ODEs as string representations."""
        pass
    
    @property
    @abstractmethod
    def input_schema(self) -> type[VPMSchema]:
        """Pydantic model defining the expected inputs (Heat, Work, Feedstock)."""
        pass
    
    @property
    @abstractmethod
    def output_schema(self) -> type[VPMSchema]:
        """Pydantic model defining the outputs (Conversion, Purity, NORM partition)."""
        pass
    
    @abstractmethod
    def build_pinn_loss(self, collocation_pts: Any) -> Any:
        """Constructs the physics-informed loss function using JAX."""
        pass
    
    @abstractmethod
    def validate(self, benchmark_data: Any) -> ValidationReport:
        """Validates the model against benchmark data and returns a report."""
        pass
