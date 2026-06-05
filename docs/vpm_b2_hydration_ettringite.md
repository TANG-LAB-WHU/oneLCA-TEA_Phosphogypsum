# B2 VPM_hydration: Hydration & Ettringite Stabilization

`VPM_hydration` is a physics-informed kinetic solver module that models the dissolution-recrystallization kinetics of phosphogypsum (PG) blended with alkaline binders (cement, lime, slag) to form interlocking network structures of **Ettringite** and **C-S-H gels** for construction applications.

---

## 1. Core Chemical Principles & Solidification Mechanism

The primary challenge of utilizing phosphogypsum in civil engineering (e.g., road base stabilization, construction blocks) is structural stability and impurity leaching. Solidification uses mineral binders to trigger recrystallization.

### 1.1 Chemical Reactions
The hydration of binder phases (like tricalcium aluminate $\text{C}_3\text{A}$) in the presence of calcium sulfate ($\text{CaSO}_4 \cdot 2\text{H}_2\text{O}$) from PG in an alkaline medium leads to the formation of **Ettringite** (Calcium Sulfoaluminate Hydrate, $\text{AFt}$):
$$3\text{Ca}^{2+} + 3\text{SO}_4^{2-} + 2\text{Al(OH)}_4^- + 4\text{OH}^- + 26\text{H}_2\text{O} \rightarrow \text{Ca}_6\text{Al}_2(\text{SO}_4)_3(\text{OH})_{12} \cdot 26\text{H}_2\text{O}$$

Ettringite crystallizes as needle-like hexagonal prisms. These needles grow and interlock, building a dense microstructural matrix that provides high early mechanical strength.

### 1.2 NORM & Heavy Metal Stabilization
*   **Isomorphic Substitution**: Heavy metals (e.g., $\text{Pb}^{2+}$, $\text{Cd}^{2+}$, $\text{Zn}^{2+}$) substitute for calcium ions ($\text{Ca}^{2+}$) in the ettringite structure.
*   **Anionic Exchange**: Harmful impurities like fluoride ($\text{F}^-$) and phosphates ($\text{PO}_4^{3-}$) substitute for sulfate groups ($\text{SO}_4^{2-}$) in the crystalline lattice.
*   **Physical Encapsulation**: The simultaneous formation of Calcium Silicate Hydrate (C-S-H) gels physically traps radioactive thorium, uranium, and radium isotopes, reducing their leaching index.

---

## 2. Microscopic Physical Hydration Model (VPM_hydration)

The hydration kinetics and exothermic phase transition are modeled in [pgloop/pathways/vpms/hydration.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/vpms/hydration.py) and constrained by three governing equations:

1.  **Hydration Phase Kinetics (Avrami-Erofeev Model)**:
    $$\frac{d\alpha}{dt} = n \cdot k \cdot t^{n-1} \cdot (1 - \alpha)$$
    *   *Physical Meaning*: Simulates the nucleation and growth of crystal nuclei. $\alpha$ represents the hydration degree (0.0 to 1.0), $k$ is the reaction rate constant, and $n$ is the Avrami exponent representing structural growth dimensions.
2.  **Exothermic Heat Conservation**:
    $$\frac{dT}{dt} = \frac{Q_{\text{hydration}} \cdot \frac{d\alpha}{dt} - U \cdot A \cdot (T - T_{\text{ambient}})}{m \cdot C_p}$$
    *   *Physical Meaning*: Models heat release since hydration is highly exothermic. High curing temperatures accelerate reaction rates but can destabilize ettringite structures.
3.  **Liquid-Solid Phase Mass Balance**:
    $$\frac{dC}{dt} = r_{\text{dissolution}}(\text{hemihydrate}) - r_{\text{crystallization}}(\text{dihydrate/ettringite})$$
    *   *Physical Meaning*: Tracks the concentration ($C$) of ions in the interstitial pore solution, balancing the dissolution of raw calcium sulfate against crystal growth consumption.

### 2.1 Schema Definition
*   **Inputs ([HydrationInputSchema](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/vpms/hydration.py#L13-L18))**:
    *   `water_solid_ratio`: Ratio of water to dry mix weight ($0.3$ to $2.0$).
    *   `temperature_c`: Initial curing temperature ($5^\circ\text{C}$ to $60^\circ\text{C}$).
    *   `retarder_dosage_ppm`: Dosing of citric/organic retarder ($0$ to $1000\text{ ppm}$).
    *   `mixing_energy_kwh`: Mixing energy consumed per tonne product.
*   **Outputs ([HydrationOutputSchema](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/vpms/hydration.py#L20-L24))**:
    *   `hydration_degree`: Final hydration conversion fraction ($0.0$ to $1.0$).
    *   `setting_time_min`: Setting/hardening duration in minutes ($2$ to $300\text{ minutes}$).
    *   `compressive_strength_mpa`: Early compressive strength evaluated at 2 hours ($1.0$ to $40.0\text{ MPa}$).

---

## 3. Macro-Scale Engineering Applications & LCA-TEA Mappings

Micro-level hydration degree ($\alpha$) and mechanical strength map directly to macro-level process performance indices:

*   **Macro Pathway Mapping**:
    *   [ConstructionMaterialsPathway](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/pg_construction.py): Models PG utilization in road bases, subgrades, backfills, and building blocks.
    *   [SoilAmendmentPathway](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/pg_soil_amendment.py): Models soil stabilization and pH adjustments.
*   **LCA Environmental Metrics**:
    *   **Global Warming Potential (GWP)**: The environmental cost is dominated by binder inputs (e.g., OPC cement clinker). The optimizer uses the model to minimize binder dosage while satisfying the target compressive strength.
    *   **Ecotoxicity & Leaching Indexes**: Tracks the leaching pollution potential of fluorine and phosphorus impurities over time.
*   **TEA Economic Performance**:
    *   **OPEX Optimization**: Raw PG is a low-cost waste. However, OPC cement binders are expensive. The decision engine optimizes the binder-to-PG ratio to keep production cost below natural aggregate benchmarks.
    *   **Aggregate Substitution Credits**: Earns economic offsets by replacing mined natural gravel aggregates with stabilized PG road-base blocks.
