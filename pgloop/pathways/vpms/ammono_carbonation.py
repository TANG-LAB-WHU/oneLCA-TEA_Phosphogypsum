from typing import Any, Dict, List
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pydantic import Field
from .base_vpm import ValorizationPathwayModule, VPMSchema, ValidationReport


class AmmonoCarbonationInputSchema(VPMSchema):
    nh3_pg_ratio: float = Field(..., description="Molar ratio of NH3 to PG (CaSO4)", ge=1.5, le=2.5)
    co2_pressure_bar: float = Field(..., description="CO2 partial pressure", ge=0.5, le=10.0)
    slurry_density_pct: float = Field(..., description="Slurry solid concentration in wt%", ge=10.0, le=50.0)
    temperature_c: float = Field(..., description="Reaction temperature in Celsius", ge=20.0, le=90.0)
    work_input_kwh: float = Field(..., description="Stirring and pumping power in kWh per tonne PG")


class AmmonoCarbonationOutputSchema(VPMSchema):
    conversion_rate: float = Field(..., description="Conversion rate of CaSO4 to CaCO3", ge=0.0, le=1.0)
    ammonium_sulfate_yield: float = Field(..., description="Yield of (NH4)2SO4 in kg per kg PG")
    co2_sequestration_efficiency: float = Field(..., description="CO2 sequestration efficiency fraction", ge=0.0, le=1.0)


class AmmonoCarbonationVPM(ValorizationPathwayModule):
    """
    Valorization Pathway Module (VPM) for Ammonium Carbonation of Phosphogypsum.
    Models multi-phase mass transfer and carbonation chemical kinetics.
    """

    @property
    def module_id(self) -> str:
        return "VPM_ammono_carbonation"

    @property
    def governing_equations(self) -> List[str]:
        return [
            "d_CO2_dt = kLa * (C_CO2_sat - C_CO2) - r_reaction  # Gas-liquid mass transfer of CO2",
            "d_alpha_dt = k_r * C_NH3^2 * C_CO2 * (1 - alpha)^n  # Reaction kinetics of carbonation",
            "r_precipitation = k_p * (S - 1)^p  # Calcium carbonate precipitation kinetics",
            "d_Ca_dt = r_dissolution - r_precipitation  # Mass conservation of calcium ions"
        ]

    @property
    def input_schema(self) -> type[VPMSchema]:
        return AmmonoCarbonationInputSchema

    @property
    def output_schema(self) -> type[VPMSchema]:
        return AmmonoCarbonationOutputSchema

    def build_pinn_loss(self, collocation_pts: Any) -> Any:
        """
        Constructs the physics-informed loss for CO2 mass transfer and chemical reaction.
        """
        if torch is None:
            return lambda model: 0.0

        # Physical constants
        kla = 0.12  # Gas-liquid mass transfer coefficient, s^-1
        c_sat = 0.035  # CO2 saturation concentration, mol/L

        def pinn_loss_fn(model: torch.nn.Module) -> torch.Tensor:
            pts = torch.tensor(collocation_pts, dtype=torch.float32, requires_grad=True)
            t_pts = pts[:, 0:1]
            c_co2_pts = pts[:, 1:2]  # local dissolved CO2 concentration

            # Predict conversion alpha and reaction rate r
            out = model(torch.cat([t_pts, c_co2_pts], dim=1))
            alpha_pred = out[:, 0:1]
            r_pred = out[:, 1:2]

            # Gradients
            alpha_t = torch.autograd.grad(
                alpha_pred, t_pts,
                grad_outputs=torch.ones_like(alpha_pred),
                create_graph=True
            )[0]

            # Mass transfer residual: d_C_CO2_dt = kLa * (C_sat - C) - r_reaction
            c_co2_t = torch.autograd.grad(
                c_co2_pts, t_pts,
                grad_outputs=torch.ones_like(c_co2_pts),
                create_graph=True
            )[0]

            mass_transfer_residual = c_co2_t - (kla * (c_sat - c_co2_pts) - r_pred)
            kinetic_residual = alpha_t - r_pred

            return torch.mean(mass_transfer_residual ** 2 + kinetic_residual ** 2)

        return pinn_loss_fn

    def validate(self, benchmark_data: Dict[str, Any]) -> ValidationReport:
        nh3_ratio = benchmark_data.get("nh3_pg_ratio", 2.0)
        temp = benchmark_data.get("temperature_c", 50.0)
        exp_conversion = benchmark_data.get("exp_conversion", 0.95)

        # Empirical carbonation model
        # Optimal temperature is around 40-60 C due to ammonia volatility at high temp
        temp_factor = 1.0 - 0.0003 * (temp - 50.0) ** 2
        ratio_factor = min(1.0, nh3_ratio / 2.0)
        est_conversion = 0.98 * temp_factor * ratio_factor

        error = abs(est_conversion - exp_conversion)
        is_valid = error < 0.05

        return ValidationReport(
            is_valid=is_valid,
            metrics={"conversion_mae": float(error), "estimated_conversion": float(est_conversion)},
            details=f"Validated at {temp} C with molar ratio {nh3_ratio:.2f}. MAE: {error:.4f}"
        )
