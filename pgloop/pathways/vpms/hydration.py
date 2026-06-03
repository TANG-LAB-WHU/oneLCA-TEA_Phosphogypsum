from typing import Any, Dict, List
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pydantic import Field
from .base_vpm import ValorizationPathwayModule, VPMSchema, ValidationReport


class HydrationInputSchema(VPMSchema):
    water_solid_ratio: float = Field(..., description="Water to solid weight ratio", ge=0.3, le=2.0)
    temperature_c: float = Field(..., description="Initial hydration temperature", ge=5.0, le=60.0)
    retarder_dosage_ppm: float = Field(..., description="Retarder concentration in ppm", ge=0.0, le=1000.0)
    mixing_energy_kwh: float = Field(..., description="Electricity consumption in mixing per tonne product")


class HydrationOutputSchema(VPMSchema):
    hydration_degree: float = Field(..., description="Final hydration degree", ge=0.0, le=1.0)
    setting_time_min: float = Field(..., description="Setting/curing time in minutes", ge=2.0, le=300.0)
    compressive_strength_mpa: float = Field(..., description="Compressive strength at 2 hours", ge=1.0, le=40.0)


class HydrationVPM(ValorizationPathwayModule):
    """
    Valorization Pathway Module (VPM) for Gypsum Hydration (re-curing).
    Models the dissolution of hemihydrate, exothermic reaction, and needle crystallization.
    """

    @property
    def module_id(self) -> str:
        return "VPM_hydration"

    @property
    def governing_equations(self) -> List[str]:
        return [
            "d_alpha_dt = n * k * t^(n-1) * (1 - alpha)  # Hydration kinetics (Avrami-Erofeev model)",
            "dT_dt = (Q_hydration * d_alpha_dt - U * A * (T - T_ambient)) / (m * Cp)  # Exothermic heat balance",
            "d_C_dt = r_dissolution(hemihydrate) - r_crystallization(dihydrate)  # Liquid phase mass conservation"
        ]

    @property
    def input_schema(self) -> type[VPMSchema]:
        return HydrationInputSchema

    @property
    def output_schema(self) -> type[VPMSchema]:
        return HydrationOutputSchema

    def build_pinn_loss(self, collocation_pts: Any) -> Any:
        """
        Constructs the physics-informed loss for Avrami-Erofeev hydration kinetics.
        """
        if torch is None:
            return lambda model: 0.0

        # Avrami constants
        n_avrami = 2.0
        k_avrami = 1e-4

        def pinn_loss_fn(model: torch.nn.Module) -> torch.Tensor:
            pts = torch.tensor(collocation_pts, dtype=torch.float32, requires_grad=True)
            t_pts = pts[:, 0:1]  # time in seconds

            # Predict hydration degree alpha
            alpha_pred = model(t_pts)

            # Gradients
            alpha_t = torch.autograd.grad(
                alpha_pred, t_pts,
                grad_outputs=torch.ones_like(alpha_pred),
                create_graph=True
            )[0]

            # Avrami rate law: d_alpha_dt - n * k * t^(n-1) * (1 - alpha)
            rate_law = n_avrami * k_avrami * (t_pts ** (n_avrami - 1)) * (1.0 - alpha_pred)
            residual = alpha_t - rate_law
            return torch.mean(residual ** 2)

        return pinn_loss_fn

    def validate(self, benchmark_data: Dict[str, Any]) -> ValidationReport:
        w_s = benchmark_data.get("water_solid_ratio", 0.6)
        retarder = benchmark_data.get("retarder_dosage_ppm", 0.0)
        exp_setting_time = benchmark_data.get("exp_setting_time", 25.0)

        # Empirical setting time model
        # Higher water-solid ratio increases setting time; retarder increases it significantly
        base_time = 15.0 + 10.0 * w_s
        retarder_effect = retarder * 0.15
        est_setting_time = base_time + retarder_effect

        error = abs(est_setting_time - exp_setting_time)
        is_valid = error < 5.0

        return ValidationReport(
            is_valid=is_valid,
            metrics={"setting_time_mae": float(error), "estimated_setting_time": float(est_setting_time)},
            details=f"Validated with W/S {w_s:.2f} and {retarder:.0f} ppm retarder. MAE: {error:.4f} min"
        )
