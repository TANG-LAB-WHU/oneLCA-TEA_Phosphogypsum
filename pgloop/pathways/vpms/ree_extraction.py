from typing import Any, Dict, List
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pydantic import Field
from .base_vpm import ValorizationPathwayModule, VPMSchema, ValidationReport


class REEExtractionInputSchema(VPMSchema):
    acid_type: str = Field(..., description="Acid type used for leaching (e.g., H2SO4, HNO3, HCl)")
    acid_concentration_m: float = Field(..., description="Acid molar concentration", ge=0.1, le=5.0)
    solid_liquid_ratio: float = Field(..., description="Solid to liquid weight ratio", ge=0.05, le=0.5)
    temperature_c: float = Field(..., description="Leaching temperature", ge=20.0, le=95.0)
    leaching_time_min: float = Field(..., description="Leaching reaction duration in minutes", ge=10.0, le=240.0)


class REEExtractionOutputSchema(VPMSchema):
    ree_recovery_pct: float = Field(..., description="Percentage of REE recovered in liquid phase", ge=0.0, le=100.0)
    acid_consumption_kg_per_t: float = Field(..., description="Acid consumption in kg per tonne of PG")
    calcium_loss_pct: float = Field(..., description="Percentage of calcium matrix co-dissolved", ge=0.0, le=100.0)


class REEExtractionVPM(ValorizationPathwayModule):
    """
    Valorization Pathway Module (VPM) for Rare Earth Element (REE) Acid Leaching.
    Models diffusion-controlled core dissolution and chemical leaching kinetics.
    """

    @property
    def module_id(self) -> str:
        return "VPM_ree_extraction"

    @property
    def governing_equations(self) -> List[str]:
        return [
            "1 - 3 * (1 - alpha)^(2/3) + 2 * (1 - alpha) = k_d * t  # Shrinking Core Model (diffusion control)",
            "d_C_acid_dt = -r_acid_consumption - D_eff * d2_C_acid_dx2  # Acid concentration diffusion",
            "d_ree_liq_dt = r_leaching - Q_out * C_ree  # Mass conservation of dissolved REEs"
        ]

    @property
    def input_schema(self) -> type[VPMSchema]:
        return REEExtractionInputSchema

    @property
    def output_schema(self) -> type[VPMSchema]:
        return REEExtractionOutputSchema

    def build_pinn_loss(self, collocation_pts: Any) -> Any:
        """
        Constructs the physics-informed loss for the Shrinking Core model.
        """
        if torch is None:
            return lambda model: 0.0

        # Diffusion kinetic rate constant
        kd_val = 0.002

        def pinn_loss_fn(model: torch.nn.Module) -> torch.Tensor:
            pts = torch.tensor(collocation_pts, dtype=torch.float32, requires_grad=True)
            t_pts = pts[:, 0:1]  # time in seconds

            # Predict extraction fraction alpha
            alpha_pred = model(t_pts)

            # Shrinking core expression: F(alpha) = 1 - 3*(1-alpha)^(2/3) + 2*(1-alpha)
            # Left side
            one_minus_alpha = torch.clamp(1.0 - alpha_pred, min=1e-6)
            f_alpha = 1.0 - 3.0 * (one_minus_alpha ** (2.0 / 3.0)) + 2.0 * one_minus_alpha

            # Residual: F(alpha) - kd * t
            residual = f_alpha - kd_val * t_pts
            return torch.mean(residual ** 2)

        return pinn_loss_fn

    def validate(self, benchmark_data: Dict[str, Any]) -> ValidationReport:
        conc = benchmark_data.get("acid_concentration_m", 1.5)
        temp = benchmark_data.get("temperature_c", 60.0)
        time = benchmark_data.get("leaching_time_min", 120.0)
        exp_recovery = benchmark_data.get("exp_recovery_pct", 65.0)

        # Empirical leaching recovery model
        # Optimal temperature is around 60-80 C, concentration around 1.5-2.0 M
        temp_factor = 1.0 - 0.0002 * (temp - 70.0) ** 2
        conc_factor = 1.0 - 0.08 * (conc - 1.8) ** 2
        est_recovery = 75.0 * temp_factor * conc_factor * (1.0 - np.exp(-0.02 * time))
        est_recovery = min(100.0, max(0.0, est_recovery))

        error = abs(est_recovery - exp_recovery)
        is_valid = error < 10.0

        return ValidationReport(
            is_valid=is_valid,
            metrics={"recovery_mae_pct": float(error), "estimated_recovery_pct": float(est_recovery)},
            details=f"Validated at {temp} C with {conc:.2f} M acid. MAE: {error:.4f}%"
        )
