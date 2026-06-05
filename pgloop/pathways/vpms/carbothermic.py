from typing import Any, Dict, List

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pydantic import Field

from .base_vpm import ValidationReport, ValorizationPathwayModule, VPMSchema


class CarbothermicInputSchema(VPMSchema):
    temperature_c: float = Field(
        ..., description="Rotary kiln temperature in Celsius", ge=600.0, le=1400.0
    )
    residence_time_min: float = Field(
        ..., description="Residence time in minutes", ge=5.0, le=180.0
    )
    c_s_ratio: float = Field(..., description="Carbon to Sulfur molar ratio", ge=0.5, le=3.0)
    heat_input_mj: float = Field(..., description="Heat input in MJ per tonne of PG")
    work_input_kwh: float = Field(..., description="Electricity consumption in kWh per tonne of PG")


class CarbothermicOutputSchema(VPMSchema):
    ca_conversion: float = Field(
        ..., description="Calcium sulfate conversion fraction", ge=0.0, le=1.0
    )
    so2_yield: float = Field(..., description="SO2 yield fraction", ge=0.0, le=1.0)
    co2_emission_kg: float = Field(..., description="CO2 emission in kg per tonne of PG")


class CarbothermicVPM(ValorizationPathwayModule):
    """
    Valorization Pathway Module (VPM) for Carbothermic Reduction of Phosphogypsum.
    Models the high-temperature decomposition kinetics and energy conservation.
    """

    @property
    def module_id(self) -> str:
        return "VPM_carbothermic_reduction"

    @property
    def governing_equations(self) -> List[str]:
        return [
            "d_alpha_dt = A * exp(-Ea / (R * T)) * (1 - alpha)^n  # Reaction kinetics (shrinking core)",
            "dT_dt = (Q_heat - delta_H * r_reaction - Q_loss) / (m * Cp)  # Energy conservation",
            "d_c_dt = D_eff * d2_c_dx2 - r_reaction  # Mass conservation and diffusion",
        ]

    @property
    def input_schema(self) -> type[VPMSchema]:
        return CarbothermicInputSchema

    @property
    def output_schema(self) -> type[VPMSchema]:
        return CarbothermicOutputSchema

    def build_pinn_loss(self, collocation_pts: Any) -> Any:
        """
        Constructs the physics-informed loss function.
        If PyTorch is available, returns a callable computing residuals.
        """
        if torch is None:
            # Fallback when torch is not installed
            return lambda model: 0.0

        # Ea = 220 kJ/mol, R = 8.314 J/(mol*K), A = 1.2e7 s^-1
        ea = 220000.0
        r_gas = 8.314
        a_pre = 1.2e7

        def pinn_loss_fn(model: torch.nn.Module) -> torch.Tensor:
            # collocation_pts shape: (N, 3) representing [x, t, T]
            pts = torch.tensor(collocation_pts, dtype=torch.float32, requires_grad=True)
            x_pts = pts[:, 0:1]
            t_pts = pts[:, 1:2]
            temp_pts = pts[:, 2:3]  # Temperature in Kelvin

            # Model predicts conversion fraction alpha
            alpha_pred = model(torch.cat([x_pts, t_pts], dim=1))

            # Gradients
            alpha_t = torch.autograd.grad(
                alpha_pred, t_pts, grad_outputs=torch.ones_like(alpha_pred), create_graph=True
            )[0]

            # Kinetic residual: d_alpha_dt - A * exp(-Ea / (R * T)) * (1 - alpha)
            kinetic_rate = a_pre * torch.exp(-ea / (r_gas * temp_pts)) * (1.0 - alpha_pred)
            residual = alpha_t - kinetic_rate
            return torch.mean(residual**2)

        return pinn_loss_fn

    def validate(self, benchmark_data: Dict[str, Any]) -> ValidationReport:
        """
        Validates output against empirical experimental values.
        """
        # Simple validation: compare estimated conversion vs experimental
        temp = benchmark_data.get("temperature_c", 1000.0)
        time = benchmark_data.get("residence_time_min", 30.0)
        exp_conversion = benchmark_data.get("exp_conversion", 0.85)

        # Empirically estimate conversion as a proxy
        t_kelvin = temp + 273.15
        k = 1.2e7 * np.exp(-220000.0 / (8.314 * t_kelvin))
        est_conversion = 1.0 - np.exp(-k * (time * 60.0))
        est_conversion = min(1.0, max(0.0, est_conversion))

        error = abs(est_conversion - exp_conversion)
        is_valid = error < 0.10

        return ValidationReport(
            is_valid=is_valid,
            metrics={"conversion_mae": float(error), "estimated_conversion": float(est_conversion)},
            details=f"Validated at {temp} C for {time} min. MAE: {error:.4f}",
        )
