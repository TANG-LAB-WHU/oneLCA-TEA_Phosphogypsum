from typing import Any, Dict, List


class BenefitCompensationModel:
    """
    Benefit Compensation Model for Phosphogypsum treatment pathways.
    Optimizes tipping fees, carbon credits, and government subsidies
    to compensate for high environmental benefit but low economic feasibility pathways.
    """

    def __init__(self, shadow_prices: Dict[str, float] = None):
        """
        Initialize model with shadow prices (USD per kg emission avoided).
        Default shadow prices map LCA impact categories to monetary values.
        """
        self.shadow_prices = shadow_prices or {
            "climate_change": 0.10,  # USD per kg CO2-eq ($100/t)
            "acidification": 2.50,   # USD per mol H+-eq
            "eutrophication_fresh": 15.00,  # USD per kg P-eq
            "human_toxicity_cancer": 5000.0,  # USD per CTUh
            "human_toxicity_noncancer": 1000.0,
            "particulate_matter": 200.0,  # USD per unit impact
        }

    def calculate_avoided_damage(
        self, baseline_impacts: Dict[str, float], pathway_impacts: Dict[str, float]
      ) -> Dict[str, float]:
        """
        Calculate monetized avoided environmental damage compared to a baseline (e.g., stack disposal).
        """
        avoided_costs = {}
        total_avoided = 0.0

        for category, shadow_price in self.shadow_prices.items():
            base_val = baseline_impacts.get(category, 0.0)
            path_val = pathway_impacts.get(category, 0.0)
            avoided_qty = base_val - path_val

            if avoided_qty > 0:
                monetized_benefit = avoided_qty * shadow_price
                avoided_costs[category] = monetized_benefit
                total_avoided += monetized_benefit
            else:
                avoided_costs[category] = 0.0

        avoided_costs["total_avoided_environmental_benefit"] = total_avoided
        return avoided_costs

    def optimize_compensation(
        self,
        pathway_code: str,
        clcc: float,  # Conventional life cycle cost (USD/t PG)
        revenue: float,  # Product revenue (USD/t PG)
        avoided_environmental_benefit: float,  # Monetized benefit (USD/t PG)
        target_margin: float = 5.0,  # Target profit margin (USD/t PG)
    ) -> Dict[str, Any]:
        """
        Optimize compensation structure (tipping fee, carbon tax credit, government subsidy)
        to make a pathway economically viable.
        """
        # Net cost before compensation
        # If negative, the pathway is already profitable
        net_deficit = clcc - revenue + target_margin

        if net_deficit <= 0:
            return {
                "pathway_code": pathway_code,
                "status": "Viable without compensation",
                "required_total_compensation": 0.0,
                "suggested_tipping_fee": 0.0,
                "suggested_subsidy": 0.0,
                "net_social_benefit": avoided_environmental_benefit - target_margin,
            }

        # Allocate compensation from social/environmental benefits
        # 1. Carbon credit portion (based on avoided environmental benefit)
        carbon_credit = min(net_deficit, avoided_environmental_benefit * 0.4)
        remaining_deficit = net_deficit - carbon_credit

        # 2. Tipping fee paid by the phosphogypsum producer (e.g. mine operator)
        suggested_tipping_fee = min(remaining_deficit, 15.0)  # capped at $15/t typical tipping fee
        remaining_deficit -= suggested_tipping_fee

        # 3. Government environmental subsidy
        suggested_subsidy = max(0.0, remaining_deficit)

        total_compensation = carbon_credit + suggested_tipping_fee + suggested_subsidy
        net_social_benefit = avoided_environmental_benefit - suggested_subsidy

        return {
            "pathway_code": pathway_code,
            "status": "Compensation required" if total_compensation > 0 else "Viable",
            "required_total_compensation": total_compensation,
            "suggested_carbon_credit": carbon_credit,
            "suggested_tipping_fee": suggested_tipping_fee,
            "suggested_subsidy": suggested_subsidy,
            "net_social_benefit_to_society": net_social_benefit,
            "social_roi": (avoided_environmental_benefit / suggested_subsidy) if suggested_subsidy > 0 else float("inf"),
        }
