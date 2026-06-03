from typing import Any, List
from .base_vpm import ValorizationPathwayModule, VPMSchema, ValidationReport

class AmmonoCarbonationVPM(ValorizationPathwayModule):
    @property
    def module_id(self) -> str:
        return "VPM_ammono_carbonation"
        
    @property
    def governing_equations(self) -> List[str]:
        return ["gas-liquid-solid mass transfer"]
        
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
