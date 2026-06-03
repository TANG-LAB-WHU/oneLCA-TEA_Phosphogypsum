from typing import Any, List
from .base_vpm import ValorizationPathwayModule, VPMSchema, ValidationReport

class REEExtractionVPM(ValorizationPathwayModule):
    @property
    def module_id(self) -> str:
        return "VPM_ree_extraction"
        
    @property
    def governing_equations(self) -> List[str]:
        return ["shrinking-core leaching equilibrium"]
        
    @property
    def input_schema(self) -> type[VPMSchema]:
        return VPMSchema
        
    @property
    def output_schema(self) -> type[VPMSchema]:
        return VPMSchema
        
    def build_pinn_loss(self, collocation_pts: Any) -> Any:
        pass
        
    def validate(self, benchmark_data: Any) -> ValidationReport:
        pass
