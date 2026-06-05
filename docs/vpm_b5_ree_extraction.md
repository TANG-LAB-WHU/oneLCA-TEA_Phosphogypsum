# B5 VPM_ree_extraction: Rare Earth Element (REE) Acid Leaching

`VPM_ree_extraction` is a physics-informed kinetic solver module that models the diffusion-controlled leaching kinetics of rare earth elements (REEs, such as $\text{La}$, $\text{Ce}$, $\text{Nd}$, $\text{Y}$) from phosphogypsum (PG) using mineral acid solutions.

---

## 1. Core Chemical Principles & Reaction Mechanism

During the wet-process production of phosphoric acid, approximately **$70\%$ to $85\%$** of the rare earth elements originally present in phosphate rock partition into the phosphogypsum waste, substituting for calcium ions in the dihydrate gypsum lattice.

### 1.1 Chemical Dissolution & Leaching
To extract these trace REEs (typically present at concentrations of $100$ to $2000\text{ ppm}$), PG is leached with inorganic acids (such as sulfuric acid $\text{H}_2\text{SO}_4$, nitric acid $\text{HNO}_3$, or hydrochloric acid $\text{HCl}$):
$$\text{CaSO}_4 \cdot 2\text{H}_2\text{O (solid containing REE)} + \text{H}^+ \xrightarrow{\text{dissolution}} \text{Ca}^{2+} + \text{SO}_4^{2-} + \text{REE}^{3+}_{\text{(aq)}} + \text{H}_2\text{O}$$

*   **Selective Dissolution**: The leaching agent must dissolve the localized REEs trapped in the crystal lattice while minimizing the mass co-dissolution of the bulk dihydrate calcium sulfate matrix (calcium loss), which consumes acid and complicates downstream separation.

### 1.2 Downstream Recovery
The pregnant leach solution (PLS) containing dissolved $\text{REE}^{3+}$ is processed via solvent extraction, ion exchange, or chemical precipitation (using oxalic acid or sodium hydroxide) to recover high-purity rare earth oxides.

---

## 2. Microscopic Physical Leaching Model (VPM_ree_extraction)

The core-diffusion mass transfer and chemical kinetics are modeled in [pgloop/pathways/vpms/ree_extraction.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/vpms/ree_extraction.py) and constrained by three governing equations:

1.  **Shrinking Core Model (Diffusion Control)**:
    $$1 - 3 \cdot (1 - \alpha)^{2/3} + 2 \cdot (1 - \alpha) = k_d \cdot t$$
    *   *Physical Meaning*: Simulates diffusion-limited leaching from spherical PG particles. $\alpha$ is the REE recovery fraction, and $k_d$ is the effective diffusion kinetic rate constant.
2.  **Acid Mass Diffusion**:
    $$\frac{\partial C_{\text{acid}}}{\partial t} = -r_{\text{acid\_consumption}} - D_{\text{eff}} \cdot \frac{\partial^2 C_{\text{acid}}}{\partial x^2}$$
    *   *Physical Meaning*: Models the consumption and diffusion ($D_{\text{eff}}$) of acid ions ($\text{H}^+$) through the solid particle shell and liquid boundary layers.
3.  **Solute Ion Mass Conservation**:
    $$\frac{dC_{\text{ree},\text{liq}}}{dt} = r_{\text{leaching}} - Q_{\text{out}} \cdot C_{\text{ree}}$$
    *   *Physical Meaning*: Conserves the mass of dissolved REEs in the liquid phase inside the leaching tank reactor.

### 2.1 Schema Definition
*   **Inputs ([REEExtractionInputSchema](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/vpms/ree_extraction.py#L13-L19))**:
    *   `acid_type`: Acid reagent used for leaching (e.g., `H2SO4`, `HNO3`, `HCl`).
    *   `acid_concentration_m`: Acid molarity in mol/L ($0.1$ to $5.0\text{ M}$).
    *   `solid_liquid_ratio`: Solid-to-liquid weight ratio ($0.05$ to $0.5$).
    *   `temperature_c`: Leaching operating temperature ($20^\circ\text{C}$ to $95^\circ\text{C}$).
    *   `leaching_time_min`: Curing duration in minutes ($10$ to $240\text{ minutes}$).
*   **Outputs ([REEExtractionOutputSchema](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/vpms/ree_extraction.py#L21-L25))**:
    *   `ree_recovery_pct`: Percentage recovery fraction of REEs dissolved in the liquid phase ($0\%$ to $100\%$).
    *   `acid_consumption_kg_per_t`: Total weight of acid consumed per tonne PG.
    *   `calcium_loss_pct`: Percentage dissolution fraction of the main calcium matrix.

---

## 3. Macro-Scale Engineering Applications & LCA-TEA Mappings

Micro-level leaching rates and acid consumption parameters scale directly to macro-level process performance indices:

*   **Macro Pathway Mapping**:
    *   [REEExtractionPathway](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/pg_ree_extraction.py): Models the downstream purification cascade, solvent extraction stages, and output rare earth oxide revenues.
*   **LCA Environmental Metrics**:
    *   **Acid Neutralization Footprint**: Spent acidic residues must be neutralized with lime ($\text{Ca(OH)}_2$). The LCA engine tracks and penalizes carbon emissions from neutralizing chemical streams.
    *   **Resource Depletion Credits**: Credits the process for recovering critical raw materials (REEs), reducing global dependence on primary open-pit mining.
*   **TEA Economic Performance**:
    *   **Strategic Revenue**: Rare earths are high-value products. Recovering La/Nd offsets leaching operational costs.
    *   **Acid Consumption Costs**: Acid consumption is a major operational cost. The optimizer uses the model to find the optimal temperature ($60^\circ\text{C}$ to $80^\circ\text{C}$) and concentration ($1.5$ to $2.0\text{ M}$) to maximize REE recovery while keeping calcium matrix dissolution low.
