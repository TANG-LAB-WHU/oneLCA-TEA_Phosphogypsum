# B1 VPM_carbothermic_reduction: Carbothermic Reduction of Phosphogypsum

`VPM_carbothermic_reduction` is a physics-informed kinetic solver module that models the high-temperature decomposition of phosphogypsum (PG) under reducing conditions (using coal or carbon inputs) to recover sulfur dioxide ($\text{SO}_2$) and calcium oxide ($\text{CaO}$) clinker.

---

## 1. Core Chemical Principles & Reaction Mechanism

The thermochemical decomposition of phosphogypsum via carbothermic reduction involves a series of solid-solid and gas-solid reactions at temperatures ranging from **$900^\circ\text{C}$ to $1300^\circ\text{C}$**.

### 1.1 Chemical Reactions
The overall reaction represents the decomposition of calcium sulfate ($\text{CaSO}_4$) in the presence of carbon:
$$\text{CaSO}_4 + 0.5\text{C} \rightarrow \text{CaO} + \text{SO}_2 + 0.5\text{CO}_2$$

This multi-step reaction occurs via intermediate pathways, including the reduction to calcium sulfide ($\text{CaS}$) followed by solid-solid reaction between sulfate and sulfide:
$$\text{CaSO}_4 + 2\text{C} \rightarrow \text{CaS} + 2\text{CO}_2$$
$$3\text{CaSO}_4 + \text{CaS} \rightarrow 4\text{CaO} + 4\text{SO}_2$$

An excess of carbon or poor oxygen control leads to the undesired accumulation of residual $\text{CaS}$, which degrades clinker quality and must be minimized by controlling the carbon-to-sulfur molar ratio ($\text{C/S}$).

### 1.2 NORM & Impurity Partitioning
*   **Volatile Phase**: Volatile heavy metals (e.g., $\text{As}$, $\text{Pb}$, $\text{Cd}$) partition partially into the flue gas phase, requiring scrubbing and purification.
*   **Solid Residue**: Refractory oxides and naturally occurring radioactive materials (NORM, e.g., $\text{Ra-226}$) concentrate in the solid $\text{CaO}$-rich ash clinker, which is subsequently stabilized by blending into cement clinker matrixes.

---

## 2. Microscopic Physical Kinetic Model (VPM_carbothermic_reduction)

The physical kinetics and mass-energy transport of the rotary kiln are modeled in [pgloop/pathways/vpms/carbothermic.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/vpms/carbothermic.py) and constrained by three governing equations:

1.  **Decomposition Kinetics (Shrinking Core Model)**:
    $$\frac{d\alpha}{dt} = A \cdot \exp\left(-\frac{E_a}{R T}\right) \cdot (1 - \alpha)^n$$
    *   *Physical Meaning*: Models the conversion rate ($\alpha$) of the solid calcium sulfate core inside the PG particle. $E_a$ is the activation energy ($\sim 220\text{ kJ/mol}$), and $A$ is the pre-exponential factor ($\sim 1.2 \times 10^7\text{ s}^{-1}$).
2.  **Exothermic/Endothermic Heat Conservation**:
    $$\frac{dT}{dt} = \frac{Q_{\text{heat}} - \Delta H_{\text{reaction}} \cdot r_{\text{reaction}} - Q_{\text{loss}}}{m \cdot C_p}$$
    *   *Physical Meaning*: Tracks temperature changes inside the rotary kiln reactor, balancing external thermal energy inputs ($Q_{\text{heat}}$) against the highly endothermic heat of reaction ($\Delta H_{\text{reaction}}$).
3.  **Local Mass Diffusion Conservation**:
    $$\frac{\partial C}{\partial t} = D_{\text{eff}} \cdot \frac{\partial^2 C}{\partial x^2} - r_{\text{reaction}}$$
    *   *Physical Meaning*: Models the diffusion of reducing gases (like $\text{CO}$) through the product layer shell of the PG particles.

### 2.1 Schema Definition
*   **Inputs ([CarbothermicInputSchema](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/vpms/carbothermic.py#L13-L18))**:
    *   `temperature_c`: Kiln operating temperature ($600^\circ\text{C}$ to $1400^\circ\text{C}$).
    *   `residence_time_min`: Kiln residence time ($5$ to $180\text{ minutes}$).
    *   `c_s_ratio`: Molar ratio of carbon to sulfur ($0.5$ to $3.0$).
    *   `heat_input_mj`: Process heat input in MJ/t PG.
    *   `work_input_kwh`: Mixing/auxiliary electricity in kWh/t PG.
*   **Outputs ([CarbothermicOutputSchema](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/vpms/carbothermic.py#L21-L25))**:
    *   `ca_conversion`: Conversion fraction of calcium sulfate ($0.0$ to $1.0$).
    *   `so2_yield`: Fraction of sulfur recovered as gaseous $\text{SO}_2$ ($0.0$ to $1.0$).
    *   `co2_emission_kg`: Carbon dioxide generated from coal heating and reduction kinetics per tonne PG.

---

## 3. Macro-Scale Engineering Applications & LCA-TEA Mappings

Micro-level conversion rates ($\alpha$) and temperature profiles ($T$) scale directly to macro-level process performance indices:

*   **Macro Pathway Mapping**:
    *   [SulfurAcidPathway](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/pg_sulfur_acid.py): Models the conversion of $\text{SO}_2$ gaseous output into commercial-grade sulfuric acid ($\text{H}_2\text{SO}_4$) and recovery of elemental sulfur.
    *   [CementPathway](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/pg_cement.py): Models the integration of $\text{CaO}$ residue clinker into industrial OPC cement mixes.
*   **LCA Environmental Metrics**:
    *   **Acidification Potential (AP)**: Gaseous $\text{SO}_2$ leakage must be strictly mitigated. The model links kinetic conversion efficiency directly to tail gas scrubber loads to prevent local acid rain impacts.
    *   **Global Warming Potential (GWP)**: High-temperature calcination is energy-intensive. The LCA engine balances fossil-fuel combustion emissions against avoided raw sulfur mining offsets.
*   **TEA Economic Performance**:
    *   **CAPEX & OPEX**: High CAPEX due to specialized rotary kilns and gas purification units. The economic engine balances raw coal/coke costs against revenues from sulfuric acid, elemental sulfur, and clinker credits.
