# B3 VPM_ammono_carbonation: Ammonium Carbonation (Merseburg Process)

`VPM_ammono_carbonation` is a physics-informed kinetic solver module that models the multi-phase reaction of phosphogypsum (PG) with ammonia ($\text{NH}_3$) and carbon dioxide ($\text{CO}_2$) to sequestrate carbon and yield ammonium sulfate [$(\text{NH}_4)_2\text{SO}_4$] fertilizer and calcium carbonate ($\text{CaCO}_3$) fillers.

---

## 1. Core Chemical Principles & Reaction Mechanism

The Ammono-Carbonation (historically the Merseburg process) represents a dual-benefit pathway: it sequestrates greenhouse gas $\text{CO}_2$ while recycling industrial sulfate wastes.

### 1.1 Chemical Reactions
The overall reaction represents the conversion of calcium sulfate dihydrate ($\text{CaSO}_4 \cdot 2\text{H}_2\text{O}$) into calcium carbonate and ammonium sulfate:
$$\text{CaSO}_4 \cdot 2\text{H}_2\text{O} + 2\text{NH}_3 + \text{CO}_2 \rightarrow \text{CaCO}_3 \downarrow + (\text{NH}_4)_2\text{SO}_4 + \text{H}_2\text{O}$$

This occurs in a three-phase (gas-liquid-solid) slurry reactor:
1.  **Gas Absorption**: Gaseous $\text{NH}_3$ and $\text{CO}_2$ dissolve into the aqueous slurry phase, forming ammonium carbonate $[(\text{NH}_4)_2\text{CO}_3]$.
2.  **Dissolution**: Calcium sulfate dihydrate dissolves to release $\text{Ca}^{2+}$ and $\text{SO}_4^{2-}$ ions.
3.  **Precipitation**: Dissolved calcium ions react with carbonate ions ($\text{CO}_3^{2-}$) to precipitate out as highly insoluble calcium carbonate ($\text{CaCO}_3$).

### 1.2 Impurity Partitioning
*   **Fertilizer Phase**: The soluble ammonium sulfate is crystallized as nitrogen fertilizer. Heavy metals must be kept below strict agricultural thresholds.
*   **Solid Calcium Carbonate**: Over 90% of the radioactive radium (Ra-226) and heavy metals co-precipitate into the $\text{CaCO}_3$ filter cake phase, which requires stabilization or washing before industrial utilization (e.g., as building fillers).

---

## 2. Microscopic Physical Mass Transfer & Reaction Model (VPM_ammono_carbonation)

The gas-liquid mass transfer and chemical precipitation kinetics are modeled in [pgloop/pathways/vpms/ammono_carbonation.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/vpms/ammono_carbonation.py) and constrained by four governing equations:

1.  **Gas-Liquid Mass Transfer ($\text{CO}_2$ Dissolution)**:
    $$\frac{dC_{\text{CO}_2}}{dt} = k_L a \cdot (C_{\text{CO}_2,\text{sat}} - C_{\text{CO}_2}) - r_{\text{reaction}}$$
    *   *Physical Meaning*: Models the rate of gaseous carbon dioxide dissolving into the aqueous slurry phase. $k_L a$ is the volumetric mass transfer coefficient, and $C_{\text{CO}_2,\text{sat}}$ is the saturation concentration of dissolved carbon dioxide.
2.  **Carbonation Reaction Kinetics**:
    $$\frac{d\alpha}{dt} = k_r \cdot C_{\text{NH}_3}^2 \cdot C_{\text{CO}_2} \cdot (1 - \alpha)^n$$
    *   *Physical Meaning*: Models the chemical reaction rate producing calcium carbonate. The rate depends quadratically on ammonia concentration and linearly on dissolved $\text{CO}_2$.
3.  **Precipitation Kinetics**:
    $$r_{\text{precipitation}} = k_p \cdot (S - 1)^p$$
    *   *Physical Meaning*: Simulates the crystallization rate of calcium carbonate based on the supersaturation index ($S$) of the solution.
4.  **Aqueous Ion Conservation**:
    $$\frac{dC_{\text{Ca}}}{dt} = r_{\text{dissolution}} - r_{\text{precipitation}}$$
    *   *Physical Meaning*: Enforces mass conservation for calcium ions ($\text{Ca}^{2+}$) in the aqueous phase.

### 2.1 Schema Definition
*   **Inputs ([AmmonoCarbonationInputSchema](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/vpms/ammono_carbonation.py#L13-L19))**:
    *   `nh3_pg_ratio`: Molar ratio of input ammonia to CaSO4 ($1.5$ to $2.5$).
    *   `co2_pressure_bar`: Carbon dioxide partial pressure in bar ($0.5$ to $10.0$).
    *   `slurry_density_pct`: Solid content in the water slurry in wt% ($10.0\%$ to $50.0\%$).
    *   `temperature_c`: Reactor temperature ($20^\circ\text{C}$ to $90^\circ\text{C}$).
    *   `work_input_kwh`: Mixing and slurry pumping power in kWh/t PG.
*   **Outputs ([AmmonoCarbonationOutputSchema](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/vpms/ammono_carbonation.py#L21-L25))**:
    *   `conversion_rate`: Final conversion fraction of CaSO4 to CaCO3 ($0.0$ to $1.0$).
    *   `ammonium_sulfate_yield`: Output yield of dry crystalline ammonium sulfate per kg PG.
    *   `co2_sequestration_efficiency`: Fraction of input $\text{CO}_2$ successfully mineralized in the solid phase.

---

## 3. Macro-Scale Engineering Applications & LCA-TEA Mappings

Micro-level conversion rates ($\alpha$) and gas absorption rates scale directly to macro-level process performance indices:

*   **Macro Pathway Mapping**:
    *   [ChemicalRecoveryPathway](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/pg_chemical_recovery.py): Evaluates the industrial recovery of agricultural fertilizers and chemical raw materials.
*   **LCA Environmental Metrics**:
    *   **Global Warming Potential (GWP)**: Calculated as a net negative carbon sink because gaseous $\text{CO}_2$ is permanently mineralized into calcium carbonate. The LCA engine credits the process for carbon sequestration.
    *   **Ammonia Slippage**: Tracks and penalizes ammonia emissions in tail gas. High reaction temperatures lead to ammonia volatility, which increase eutrophication indicators.
*   **TEA Economic Performance**:
    *   **OPEX Trade-Offs**: Raw ammonia input is a major operational cost. The optimizer solves for the optimal reaction temperature ($40^\circ\text{C}$ to $60^\circ\text{C}$) to maximize conversion while preventing expensive ammonia evaporation loss.
    *   **Product Revenue**: Credits the process based on market values of ammonium sulfate fertilizers and calcium carbonate fillers.
