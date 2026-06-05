# Life Cycle Assessment (LCA) Framework

PhosphogypsumBot implements a rigorous Life Cycle Assessment (LCA) engine following the **ISO 14040/14044** international standards for environmental management.

---

## 📐 Goal and Scope Definition

The primary objective of the LCA module is to quantify the cradle-to-gate environmental footprints of various industrial phosphogypsum (PG) treatment and valorization pathways.

### Functional Unit
*   **Definition**: **1 tonne of raw phosphogypsum treated** ($1000\text{ kg}$).
*   All raw materials, thermal energy inputs (heat/steam duties), work inputs (electricity), process yields, and environmental emissions are dynamically scaled to this functional unit (FU) inside `LCAEngine.calculate()`.

---

## 📊 10 Environmental Impact Categories

The `LCAEngine` evaluates environmental performance across 10 standardized midpoint impact categories:

| Category Code | Impact Category | Reference Unit |
| :--- | :--- | :--- |
| `climate_change` | Global Warming Potential (GWP) | $\text{kg CO}_2\text{-eq}$ |
| `acidification` | Terrestrial Acidification | $\text{mol H}^+\text{-eq}$ |
| `eutrophication_fresh` | Freshwater Eutrophication | $\text{kg P-eq}$ |
| `eutrophication_marine` | Marine Eutrophication | $\text{kg N-eq}$ |
| `human_toxicity_cancer` | Human Toxicity (Cancer effects) | $\text{CTUh}$ (Comparative Toxic Unit for humans) |
| `human_toxicity_noncancer` | Human Toxicity (Non-cancer effects) | $\text{CTUh}$ |
| `ecotoxicity_freshwater` | Freshwater Ecotoxicity | $\text{CTUe}$ (Comparative Toxic Unit for ecosystems) |
| `ionizing_radiation` | Ionizing Radiation | $\text{kBq U-235 eq}$ |
| `particulate_matter` | Particulate Matter | $\text{disease incidence}$ |
| `resource_depletion` | Abiotic Resource Depletion | $\text{kg Sb-eq}$ (Antimony equivalents) |

---

## 🔄 Life Cycle Inventory (LCI) scaling

For any valorization pathway, the process flowsheet defines input and output inventories:
1.  **Technosphere Inputs**: Direct process requirements such as:
    *   *Thermal energy* (coal, natural gas, steam)
    *   *Electric power* (kiln rotation, pump duty, agitation)
    *   *Chemical reagents* (sulfuric acid, ammonium hydroxide, organic solvents)
2.  **Biosphere Outputs**: Direct emissions to the atmosphere ($CO_2$, $SO_2$, $NO_x$), hydrosphere (heavy metal runoff, acidity), and lithosphere (partitioned radioactive elements like $^{226}Ra$).

The inventory values are fetched from the pathway definitions and multiplied by their specific characterization factors ($CF$):

$$\text{Impact}_i = \sum_{j} \text{Inventory}_j \times CF_{i,j}$$

Where $\text{Impact}_i$ is the environmental footprint in impact category $i$, and $CF_{i,j}$ is the characterization factor of inventory flow $j$ for category $i$.

---

## 🎲 Monte Carlo Uncertainty Propagation

Since input parameters (such as kiln efficiency, raw materials, grid mixes, and emissions) have inherent variances, the `LCAEngine` executes Monte Carlo simulations to propagate uncertainty:

1.  **Parameter Distributions**: Parameters are defined using standard probability distributions (e.g., Lognormal, Normal, Uniform, or Triangular) derived from experimental trials and scientific literature.
2.  **Sampling**: Samples are drawn at random over $N$ iterations (typically $N=1000$).
3.  **Result Aggregation**: The engine calculates the statistical distribution (mean, standard deviation, and 5th/50th/95th percentiles) for each impact category, which the agent uses to quantify pathway discernibility.
