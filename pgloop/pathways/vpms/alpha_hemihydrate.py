from typing import Any, Dict, List

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pydantic import Field

from .base_vpm import ValidationReport, ValorizationPathwayModule, VPMSchema


class AlphaHemihydrateInputSchema(VPMSchema):
    temperature_c: float = Field(..., description="Autoclave temperature", ge=90.0, le=160.0)
    pressure_bar: float = Field(..., description="Steam pressure in bar", ge=1.0, le=6.0)
    solid_liquid_ratio: float = Field(
        ..., description="Solid to liquid weight ratio", ge=0.1, le=1.5
    )
    additive_dosage_pct: float = Field(
        ..., description="Crystallization modifier dosage in wt%", ge=0.0, le=2.0
    )
    heat_input_mj: float = Field(..., description="Heat input in MJ per tonne of hemihydrate")


class AlphaHemihydrateOutputSchema(VPMSchema):
    hemihydrate_yield: float = Field(
        ..., description="Yield of alpha-hemihydrate phase", ge=0.0, le=1.0
    )
    aspect_ratio: float = Field(
        ..., description="Mean aspect ratio of crystals (L/D)", ge=1.0, le=20.0
    )
    purity_pct: float = Field(..., description="Hemihydrate purity percentage", ge=80.0, le=100.0)


class AlphaHemihydrateVPM(ValorizationPathwayModule):
    """
    Valorization Pathway Module (VPM) for Alpha-Hemihydrate Gypsum crystallization.
    Models phase transition dynamics and crystal morphology control.
    """

    @property
    def module_id(self) -> str:
        return "VPM_alpha_hemihydrate"

    @property
    def governing_equations(self) -> List[str]:
        return [
            "d_alpha_dt = k_c * (C - C_sat)^m  # Crystal growth rate (dissolution-recrystallization kinetics)",
            "d_L_dt = G_L * (1 - exp(-E_aspect))  # Crystal length growth rate",
            "d_D_dt = G_D * exp(-E_aspect)  # Crystal diameter growth rate",
            "d_H_dt = Q_latent * d_alpha_dt + U * A * (T_steam - T)  # Energy conservation and phase transition",
        ]

    @property
    def input_schema(self) -> type[VPMSchema]:
        return AlphaHemihydrateInputSchema

    @property
    def output_schema(self) -> type[VPMSchema]:
        return AlphaHemihydrateOutputSchema

    def build_pinn_loss(self, collocation_pts: Any) -> Any:
        """
        Constructs the physics-informed loss for dissolution-crystallization rate.
        """
        if torch is None:
            return lambda model: 0.0

        # Crystallization constants
        k_growth = 0.05
        c_sat = 0.02  # Saturation concentration

        def pinn_loss_fn(model: torch.nn.Module) -> torch.Tensor:
            pts = torch.tensor(collocation_pts, dtype=torch.float32, requires_grad=True)
            t_pts = pts[:, 0:1]
            conc_pts = pts[:, 1:2]  # concentration

            # Predict crystallization conversion alpha
            alpha_pred = model(torch.cat([t_pts, conc_pts], dim=1))

            # Gradients
            alpha_t = torch.autograd.grad(
                alpha_pred, t_pts, grad_outputs=torch.ones_like(alpha_pred), create_graph=True
            )[0]

            # Growth rate equation: d_alpha_dt - k_growth * (C - C_sat)^1.5
            rate_law = k_growth * torch.clamp(conc_pts - c_sat, min=0.0) ** 1.5
            residual = alpha_t - rate_law
            return torch.mean(residual**2)

        return pinn_loss_fn

    def validate(self, benchmark_data: Dict[str, Any]) -> ValidationReport:
        temp = benchmark_data.get("temperature_c", 120.0)
        time = benchmark_data.get("residence_time_min", 60.0)
        exp_yield = benchmark_data.get("exp_yield", 0.95)

        # Simplified empirical estimation of crystallization conversion
        # Typical optimum is around 110-130C
        optimum_factor = 1.0 - 0.0005 * (temp - 120.0) ** 2
        est_yield = min(1.0, max(0.0, 0.98 * optimum_factor * (1.0 - np.exp(-0.05 * time))))

        error = abs(est_yield - exp_yield)
        is_valid = error < 0.08

        return ValidationReport(
            is_valid=is_valid,
            metrics={"yield_mae": float(error), "estimated_yield": float(est_yield)},
            details=f"Validated at {temp} C for {time} min. MAE: {error:.4f}",
        )
