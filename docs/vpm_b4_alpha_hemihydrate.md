# B4 VPM_alpha_hemihydrate: α-Hemihydrate Calcination (Crystallization)

`VPM_alpha_hemihydrate` is a physics-informed kinetic solver module that models the hydrothermal calcination and crystallization kinetics of phosphogypsum (PG) under steam or aqueous pressure to produce high-strength $\alpha$-hemihydrate plaster ($\alpha\text{-CaSO}_4 \cdot 0.5\text{H}_2\text{O}$).

---

## 1. Core Chemical Principles & Reaction Mechanism

Phosphogypsum consists mainly of dihydrate gypsum ($\text{CaSO}_4 \cdot 2\text{H}_2\text{O}$). Heated under pressure, it loses part of its water of crystallization to form hemihydrate plaster, which exists in two forms: $\alpha$-type (high strength, dense, large crystal grains) and $\beta$-type (lower strength, porous).

### 1.1 Hydrothermal Phase Transition
The transition of dihydrate ($\text{DH}$) to $\alpha$-hemihydrate ($\alpha\text{-HH}$) occurs via a dissolution-crystallization mechanism in an autoclave or pressurized aqueous media at temperatures between **$90^\circ\text{C}$ and $160^\circ\text{C}$**:
$$\text{CaSO}_4 \cdot 2\text{H}_2\text{O (solid)} \xrightarrow{\text{dissolution}} \text{Ca}^{2+} + \text{SO}_4^{2-} + 2\text{H}_2\text{O} \xrightarrow{\text{recrystallization}} \alpha\text{-CaSO}_4 \cdot 0.5\text{H}_2\text{O (solid)} + 1.5\text{H}_2\text{O}$$

*   **Dissolution**: Dihydrate gypsum dissolves into the pressurized solution.
*   **Recrystallization**: Since the solubility of hemihydrate is lower than dihydrate at temperatures above $97^\circ\text{C}$, the solution becomes supersaturated with respect to hemihydrate, triggering its nucleation and crystallization.

### 1.2 Crystal Habit Modification
Without modifiers, $\alpha$-hemihydrate crystals grow into long needle-like or fibrous shapes with a high aspect ratio (length-to-diameter ratio, $L/D$). This results in poor water demand performance and low compacted strength. Organic crystallization modifiers (e.g., succinic acid, maleic acid) are added to adsorb selectively on specific crystal faces, suppressing growth along the c-axis and promoting the formation of short columnar crystals with low aspect ratios.

---

## 2. Microscopic Physical Crystallization Model (VPM_alpha_hemihydrate)

The phase transition rate and morphology parameters are modeled in [pgloop/pathways/vpms/alpha_hemihydrate.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/vpms/alpha_hemihydrate.py) and constrained by four governing equations:

1.  **Crystallization Growth Kinetics**:
    $$\frac{d\alpha}{dt} = k_c \cdot (C - C_{\text{sat}})^m$$
    *   *Physical Meaning*: Models the rate of crystal growth based on solution supersaturation ($C - C_{\text{sat}}$). $\alpha$ represents the phase conversion fraction, $k_c$ is the crystallization constant, and $m$ is the kinetic growth exponent.
2.  **Crystal Length Growth Rate**:
    $$\frac{dL}{dt} = G_L \cdot (1 - \exp(-E_{\text{aspect}}))$$
    *   *Physical Meaning*: Models the growth rate along the longitudinal length ($L$) of the crystal grain.
3.  **Crystal Diameter Growth Rate**:
    $$\frac{dD}{dt} = G_D \cdot \exp(-E_{\text{aspect}})$$
    *   *Physical Meaning*: Models the growth rate along the lateral diameter ($D$). Modifying additives decrease the effective aspect ratio ($E_{\text{aspect}}$), shifting growth from length to diameter.
4.  **Energy & Phase Heat Conservation**:
    $$\frac{dH}{dt} = Q_{\text{latent}} \cdot \frac{d\alpha}{dt} + U \cdot A \cdot (T_{\text{steam}} - T)$$
    *   *Physical Meaning*: Conserves thermal energy inside the pressurized autoclave, balancing steam heat inputs against the latent heat of phase crystallization ($Q_{\text{latent}}$).

### 2.1 Schema Definition
*   **Inputs ([AlphaHemihydrateInputSchema](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/vpms/alpha_hemihydrate.py#L13-L19))**:
    *   `temperature_c`: Autoclave operating temperature ($90^\circ\text{C}$ to $160^\circ\text{C}$).
    *   `pressure_bar`: Steam or hydrostatic pressure ($1.0$ to $6.0\text{ bar}$).
    *   `solid_liquid_ratio`: Solid-to-liquid weight ratio ($0.1$ to $1.5$).
    *   `additive_dosage_pct`: Dosage of crystallization modifiers in wt% ($0.0\%$ to $2.0\%$).
    *   `heat_input_mj`: Thermal energy in MJ consumed per tonne hemihydrate.
*   **Outputs ([AlphaHemihydrateOutputSchema](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/vpms/alpha_hemihydrate.py#L21-L25))**:
    *   `hemihydrate_yield`: Phase yield fraction of alpha-hemihydrate ($0.0$ to $1.0$).
    *   `aspect_ratio`: Mean length-to-diameter aspect ratio ($L/D$) of crystals ($1.0$ to $20.0$).
    *   `purity_pct`: Percentage purity of the resulting plaster ($80\%$ to $100\%$).

---

## 3. Macro-Scale Engineering Applications & LCA-TEA Mappings

Micro-level crystal purity and shape map directly to macro-level process performance indices:

*   **Macro Pathway Mapping**:
    *   [CementPathway](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/pg_cement.py): Alpha-hemihydrate is used as high-strength engineering gypsum plaster, partition boards, or self-leveling floors.
*   **LCA Environmental Metrics**:
    *   **Global Warming Potential (GWP)**: Pressurized autoclaving consumes thermal energy. The LCA engine evaluates fossil fuel carbon emissions and compares them against avoided natural gypsum calcination baselines.
*   **TEA Economic Performance**:
    *   **Premium Quality Pricing**: Standard dihydrate gypsum has low market value. In contrast, high-purity, low-aspect-ratio $\alpha$-hemihydrate plaster commands high prices in building markets.
    *   **Chemical Additives Costs**: Crystallization modifiers are expensive. The optimizer uses the model to minimize modifier dosage while keeping aspect ratio below critical limits.
