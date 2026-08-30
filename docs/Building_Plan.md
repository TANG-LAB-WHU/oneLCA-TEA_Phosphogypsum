# PhosphogypsumBot: A Physics-Informed AI Platform for Sustainable Phosphorus Resource Cycling
## Technical Building Plan & Implementation Roadmap

> [!NOTE]
> **Project Vision Statement**
> To transform phosphogypsum (PG) utilization from empirical trial-and-error chemistry into a highly predictable, multi-objective, and physics-constrained engineering science. By combining **Physics-Informed Artificial Intelligence (PI-AI)** with **Bayesian Uncertainty Quantification (UQ)**, PhosphogypsumBot serves as an intelligent decision-support system that balances **Technology, Economy, Policy, Environment, and Society (TEPES)** to discover, evaluate, and scale revolutionary PG valorization pathways.
>
> **Implementation Status (v0.6.0)**: The core modules of the platform have been successfully built, integrated, and validated on the **Wuhan University Supercomputing Center (WHU-SCC)**.

---

## 1. Project Background & Methodological Framework

Phosphogypsum (PG) is an industrial solid waste generated during the production of phosphoric acid via the wet-process treatment of phosphate rock with sulfuric acid:
$$\text{Ca}_5(\text{PO}_4)_3\text{F} + 5\text{H}_2\text{SO}_4 + 10\text{H}_2\text{O} \rightarrow 5(\text{CaSO}_4 \cdot 2\text{H}_2\text{O}) + 3\text{H}_3\text{PO}_4 + \text{HF}$$

For every ton of phosphoric acid ($P_2\text{O}_5$) produced, approximately **4.5 to 5.5 tons of PG** are generated. These stockpiles pose severe environmental risks, including leaching of residual acids, heavy metals, and naturally occurring radioactive materials (NORM, primarily Ra-226).

To address this, our **Sustainable Resource Cycling Engineering Science** framework utilizes a closed-loop, multi-level systems approach driven by **PI-AI** to optimize five core sustainability dimensions:

```mermaid
graph TD
    subgraph Resource_Input ["External Resource Inputs"]
        U_in["Input Vector U_in [Heat, Work, Money]"]
    end

    subgraph PI_AI_Engine ["PI-AI Core Engine"]
        PINN["PINNs / Fokker-Planck solvers"]
        MCMC["MCMC Chain Samplers (UQ)"]
        Embed["Multimodal Embedding (Cross-scale)"]
    end

    subgraph MLS ["Multi-Level System Simulation"]
        Micro["Micro: Chemical Kinetics"]
        Meso["Meso: Flowsheet Process Engine"]
        Macro["Macro: LCA-TEA Engine"]
    end

    subgraph TEPES ["Sustainability Dimensions"]
        direction LR
        Tech(("Technology"))
        Econ(("Economy"))
        Soc(("Society"))
        Env(("Environment"))
        Pol(("Policy"))
        
        Tech --> Econ --> Soc --> Env --> Pol --> Tech
    end

    subgraph Decision ["Adaptive Steering & Optimization"]
        BCM["Benefit Compensation Model"]
        RDO["Reverse Design Optimizer"]
        MCDA["Pathway Ranker (TOPSIS/AHP)"]
    end

    Resource_Input --> MLS
    PINN --> MLS
    MLS --> PINN
    MCMC --> MLS
    MLS --> MCMC
    Embed --> MLS
    MLS --> Embed
    
    MLS --> TEPES
    
    TEPES --> Decision
    Decision --> Resource_Input
    Decision -.-> PI_AI_Engine
```

---

## 2. Modular System Architecture Overview

PhosphogypsumBot is designed with a **strictly modular plug-in architecture**. This ensures that the system can scale, accept new chemical pathways, and upgrade its AI engines independently without monolithic refactoring.

### 2.1 Module Registry & Interface Contracts
Every component is a "Module" that registers with a central `ModuleRegistry`. Modules communicate via rigidly defined I/O schemas. For example, any chemical process is encapsulated as a `ValorizationPathwayModule (VPM)`.

The base classes are implemented in [pgloop/pathways/vpms/base_vpm.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/vpms/base_vpm.py):

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel

class VPMSchema(BaseModel):
    """Base schema for inputs and outputs of a VPM."""
    pass

class ValidationReport(BaseModel):
    """Report generated after validating the VPM against benchmarks."""
    is_valid: bool
    metrics: Dict[str, float]
    details: str

class ValorizationPathwayModule(ABC):
    """
    Abstract interface for all Phosphogypsum treatment pathway PINN models.
    Enforces a strict modular contract for I/O and physics.
    """
    
    @property
    @abstractmethod
    def module_id(self) -> str:
        """Unique identifier for the module (e.g., 'VPM_carbothermic_reduction')."""
        pass
    
    @property  
    @abstractmethod
    def governing_equations(self) -> List[str]:
        """Returns the list of governing PDEs/ODEs as string representations."""
        pass
    
    @property
    @abstractmethod
    def input_schema(self) -> type[VPMSchema]:
        """Pydantic model defining the expected inputs (Heat, Work, Feedstock)."""
        pass
    
    @property
    @abstractmethod
    def output_schema(self) -> type[VPMSchema]:
        """Pydantic model defining the outputs (Conversion, Purity, NORM partition)."""
        pass
    
    @abstractmethod
    def build_pinn_loss(self, collocation_pts: Any) -> Any:
        """Constructs the physics-informed loss function using PyTorch/JAX."""
        pass
    
    @abstractmethod
    def validate(self, benchmark_data: Any) -> ValidationReport:
        """Validates the model against benchmark data and returns a report."""
        pass
```

### 2.2 System Integration Map
The system is divided into five logical module groups:
*   **Group A (Data Foundation):** Ingests and structures raw text, standards, and literature data into vector-graph indices.
*   **Group B (PI-AI Core):** Executes physics-constrained ML, interatomic property modeling, and Bayesian MCMC inference.
*   **Group C (Simulation Engine):** Scales predictions from micro unit operations to macroscopic LCA-TEA processes.
*   **Group D (Decision Engine):** Ranks pathways, optimizes incentive compensations, and executes reverse Bayesian design.
*   **Group E (Platform):** Orchestrates agent tasks, triggers interactive visualizations, and hosts the LLM chat interface.

---

## 3. Data Foundation Layer (Module Group A)

This layer transforms raw, heterogeneous industrial data, scientific papers, and policy manuals into structured, physics-ready inputs.

### Module A1: Data Ingestion & ETL Pipeline
*   **Implementation Location**: [pgloop/iodata/](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/iodata/)
*   **Function:** Handles document extraction and standardization.
    *   **PDF Ingestion:** Uses a multi-engine parsing architecture ([pdf_parser.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/iodata/pdf_parser.py)) featuring **IBM Docling** for high-precision table and formula extraction, alongside **MinerU** and **PyMuPDF**.
    *   **Web Scraper:** Collects regional electricity grid data, chemical prices, and environmental standards ([web_scraper.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/iodata/web_scraper.py)).
    *   **Data Standardizer:** Normalizes inputs into uniform industrial metrics ([data_standardizer.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/iodata/data_standardizer.py)).

### Module A2: Knowledge Graph & RAG Engine
*   **Implementation Location**: [pgloop/knowledge/](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/knowledge/)
*   **Function:** Builds and queries the system's ontological domain knowledge base.
    *   **Extraction:** Uses an LLM information extractor ([llm_extractor.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/knowledge/llm_extractor.py)) to map entities (`PG_Source`, `Impurity`, `Product`, `Process_Unit`, `Regulation`) and relations (`CONTAINS`, `TRANSFORMS_TO`, `CATALYZES`, `RESTRICTED_BY`).
    *   **Validation & Constraints:** Automatically evaluates extracted parameters against physical boundaries ([parameter_ranges.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/knowledge/parameter_ranges.py)) and solves missing values under thermodynamic limits ([gap_filler.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/knowledge/gap_filler.py)).
    *   **Storage & Query:** Interfaces with Neo4j ([graph/neo4j_adapter.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/knowledge/graph/neo4j_adapter.py)) and local vector-graph models (**LightRAG** / **RAGAnything** in [lightrag_engine.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/knowledge/lightrag_engine.py)).

### Module A3: Materials Database & NORM Radioactivity Tracking
*   **Implementation Location**: [pgloop/chemicals/](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/chemicals/)
*   **Function:** Integrates material property databases and incorporates a **MACE machine-learning interatomic potential** predictor to estimate physical properties (e.g., lattice structures, phase transition boundaries). Tracks mass balance and partitioning of heavy metals and radioactive isotopes (Ra-226, U-238, Th-232) across VPM processes.

---

## 4. PI-AI Core Engine (Module Group B)

The intelligence core relies on the `ValorizationPathwayModule (VPM)` interface, allowing seamless addition of new PG treatment technologies.

### 4.1-4.5 VPM Instances (The Chemical Pathways)
Located in [pgloop/pathways/vpms/](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/vpms/):

| Module ID | B1: Carbothermic Reduction | B2: Hydration & Ettringite | B3: Ammono-Carbonation (Merseburg) | B4: α-Hemihydrate Calcination | B5: REE Selective Extraction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pathway** | Thermal decomposition to SO₂ + CaO (Sulfuric Acid & Cement Co-production) | Road base stabilization | Conversion to CaCO₃ + (NH₄)₂SO₄ | High-value plaster/mold production | Strategic La/Ce/Nd recovery |
| **Macro Class** | [SulfurAcidPathway](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/pg_sulfur_acid.py) | [ConstructionPathway](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/pg_construction.py) / [SoilAmendmentPathway](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/pg_soil_amendment.py) | [ChemicalRecoveryPathway](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/pg_chemical_recovery.py) | [CementPathway](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/pg_cement.py) | [REEExtractionPathway](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/pathways/pg_ree_extraction.py) |
| **Governing PDE** | Reaction-diffusion heat transfer | Crystallization-pressure diffusion | Gas-liquid-solid mass transfer | Dissolution-recrystallization | Shrinking-core leaching kinetics |
| **NORM Tracking** | Concentrates in CaO ash | Immobilized in C-S-H gel | Partitions >90% into CaCO₃ | Retained in gypsum lattice | Tracked in pregnant leach solution |

### 4.6 Advanced PINN & Stochastic Solvers
*   **Implementation Location**: [pgloop/stochastic_dynamics/](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/stochastic_dynamics/)
*   **Function:** Solves physical transport boundaries and chemical phase density states.
    *   **Fokker-Planck Solver:** Computes multi-dimensional density distributions over time using Fokker-Planck PDE models ([fokker_planck.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/stochastic_dynamics/fokker_planck.py)).
    *   **PINN Architectures:** Integrates PyTorch neural networks optimized for solving Fokker-Planck transport boundaries ([pinn.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/stochastic_dynamics/pinn.py)).
    *   **Latent SDEs & VAEs:** Captures stochastic process anomalies and molecular dynamics ([latent_sde.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/stochastic_dynamics/latent_sde.py), [vae.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/stochastic_dynamics/vae.py)).

### 4.7 Bayesian MCMC Engine
*   **Implementation Location**: [pgloop/uncertainty/](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/uncertainty/)
*   **Function:** Quantifies parameter uncertainty (reaction rates, activation energies) and propagates process variance.
    *   **Samplers:** Custom Python implementations of **Metropolis-Hastings**, **Hamiltonian Monte Carlo**, and **Gibbs Sampler** ([chain_sampling.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/uncertainty/chain_sampling.py)) to calibrate joint parameter distributions.
    *   **Bayesian Updater:** Executes closed-loop inference updating parameter distributions based on real-time observations ([bayesian_update.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/uncertainty/bayesian_update.py)).
    *   **Sensitivity Screening:** Performs global sensitivity analysis using Sobol and Delta indices ([sensitivity.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/uncertainty/sensitivity.py)).

```python
# Conceptual framework for our Metropolis-Hastings Sampler class
class MetropolisHastings(BaseMCMC):
    """Metropolis-Hastings MCMC Sampler for parameter calibration."""
    
    def sample(self, n_samples: int, warmup: int = 1000, adapt_proposal: bool = True) -> MCMCResult:
        # Runs random-walk MCMC sampling over parameter priors, computes
        # acceptance rates based on log_prob_fn(state) and returns calibrated posterior distributions.
        ...
```

---

## 5. Multi-Level Simulation Engine (Module Group C)

This engine bridges physical scales from reactors to global economics using explicit upscaling coupling protocols.

### Module C1: Micro-Level Unit Operations
*   **Implementation Location**: [pgloop/equipment/](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/equipment/)
*   **Function:** Simulates physical reactor hardware units.
    *   Includes modeling of **CSTR Reactors**, **Leaching Tanks**, **Mixing Tanks**, and **Separation Filters**.
    *   Couples reaction kinetics (VPM outputs) directly to unit mass-energy conservation boundaries.

### Module C2: Meso-Level Flowsheet Process Engine
*   **Implementation Location**: [pgloop/simulation/](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/simulation/)
*   **Function:** Solves mass and energy balances across the plant flowsheet (`micro/` reactor outputs scaling to `meso/` factory streams). Calculates aggregate parameters (total steam $U_{heat}$, electrical energy $U_{work}$, raw material streams).

### Module C3: Macro-Level LCA-TEA Evaluator
*   **Implementation Location**: [pgloop/lca/](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/lca/) & [pgloop/tea/](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/tea/)
*   **Function:** Evaluates life-cycle environmental and techno-economic footprints.
    *   **LCAEngine:** Compiles the Life Cycle Inventory (LCI) and scales environmental indicators (including GWP, water footprint, particulate matter, NORM exposure) based on ISO 14040/14044 norms.
    *   **TEAEngine:** Calculates Capital Expenditures (CAPEX), Operational Expenditures (OPEX), revenues, and tracks conventional financial metrics (NPV, IRR, Payback Period).

---

## 6. Decision & Optimization Engine (Module Group D)

This layer converts physical and economic metrics into policy recommendations and optimal process parameters.

### Module D1: Benefit Compensation Model
*   **Implementation Location**: [pgloop/decision/benefit_compensation.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/decision/benefit_compensation.py)
*   **Function:** Optimizes financial support parameters. Evaluates avoided environmental damage based on **shadow pricing** and back-calculates required carbon tax credits, tipping fees, or direct government subsidies to make low-margin, high-benefit pathways profitable.

### Module D2: Reverse Design Optimizer
*   **Implementation Location**: [pgloop/decision/optimizer/reverse_design.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/decision/optimizer/reverse_design.py)
*   **Function:** Uses Bayesian Optimization with a **Gaussian Process Regressor** (Matern kernel) to back-calculate required process inputs (e.g., solid-liquid ratios, operating temperatures, or material purity) that satisfy target threshold constraints (e.g., GWP $\le$ 100 kg CO₂-eq/t, NPV $\ge$ \$20/t).

### Module D3: Multi-Criteria Pathway Ranker
*   **Implementation Location**: [pgloop/decision/mcda.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/decision/mcda.py) & [recommender.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/decision/recommender.py)
*   **Function:** Ranks pathways across TEPES criteria using Multi-Criteria Decision Analysis (**TOPSIS**, **AHP**, and WSM). Computes ideal and anti-ideal solution vectors to find the most balanced pathway.

### Module D4: Micro & Macro Risk Assessment
*   **Implementation Location**: [pgloop/risk/](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/risk/)
*   **Function:** Aggregates process-level (technical, operational) and regional-level (political, policy, carbon tax volatility) risks into unified safety boundaries.

---

## 7. Platform & Deployment (Module Group E)

### Module E1: Streamlit Dashboard
*   **Implementation Location**: [pgloop/visualization/dashboard.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/visualization/dashboard.py)
*   **Function:** Renders an interactive web interface. Includes 3D Pareto frontier visualizations, process parameter sliders, Monte Carlo distributions, and TEPES dimension graphs.

### Module E2: Report Exporter
*   **Implementation Location**: [pgloop/visualization/report.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/visualization/report.py)
*   **Function:** Compiles analytical outputs, MCDA rankings, and uncertainty charts into standard HTML and Excel spreadsheets for engineering reviews.

### Module E3: Autonomous Plan-and-Solve Agent (10-Tool Portfolio)
*   **Implementation Location**: [chat_agent/](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/chat_agent/)
*   **Function:** Hosts an autonomous engineering agent ([agent.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/chat_agent/agent.py)) driven by a multi-step **Plan-and-Solve Chain-of-Thought (CoT)** reasoning loop. The agent interfaces with 10 deterministic backend Python tools ([tools.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/chat_agent/tools.py)), including Bayesian Reverse Design (`optimize_reverse_design`), Benefit Compensation (`optimize_benefit_compensation`), MCMC Parameter Calibration (`calibrate_process_parameters`), Crystal Properties (`predict_crystal_properties`), Live IoT Telemetry (`query_realtime_telemetry`), Forward LCA/TEA (`calculate_lca_tea`), and 5D TEPES Ranking (`rank_all_pathways`).

### Module E4: WHU-SCC Slurm Clusters & llama.cpp Inference
*   **Implementation Location**: [slurm_jobs/](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/slurm_jobs/)
*   **Function:** Orchestrates training and inference on the Wuhan University Supercomputing Center. Standardized exclusively on **`llama.cpp` (`llama-server`)** with multi-hardware auto-routing (`a100x4` GPU, `gpu` V100, and `9a14a` AMD EPYC CPU), implementing **NUMA memory interleaving (`numactl --interleave=all`)** and automated process lifecycle traps.

---

## 8. Development Timeline & Milestones

### Historical Releases & Completed Milestones
*   **Phase 1: Foundation (v0.1.0 - v0.2.0)**
    *   *Achievements*: Authoring core mathematical models (LCA/TEA engines). Constructing Neo4j databases and parser pipelines (MinerU/PyMuPDF pdf extraction).
*   **Phase 2: Physics-Informed AI (v0.3.0)**
    *   *Achievements*: Writing Fokker-Planck solvers and VPM kinetics. Building MCMC chain samplers (Metropolis-Hastings, Hamiltonian MC) for Bayesian uncertainty.
*   **Phase 3: Decision & System Optimization (v0.5.0)**
    *   *Achievements*: Integrating the Reverse Design GP optimizer, Benefit Compensation model, and MCDA TOPSIS ranker. Deploying on WHU-SCC Slurm clusters with GPU acceleration.
*   **Phase 4: Agentic AI & 10-Tool Plan-and-Solve (v0.6.0 - Current)**
    *   *Achievements*: Expanding to full 10-tool autonomous agent portfolio, Plan-and-Solve CoT reasoning loop, IBM Docling SOTA literature ingestion, and unified llama.cpp WHU-SCC Slurm orchestration.

### Next-Phase Plans: Road to Production (v0.7.0 ~ v1.0.0)

| Focus Area | Target Version | Module Focus | Core Deliverable | Success Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **Docling Literature Scale** | v0.7.0 | Group A | High-speed batch parsing of 5,000+ solid waste papers on WHU-SCC | Complete parameter range graph across all pathways |
| **Domain SFT Fine-Tuning** | v0.8.0 | `Phosphogypsum-Qwen-32B` | LoRA fine-tuning on 50k Plan-and-Solve tool trajectories | Tool calling accuracy > 98.5% on complex chemical prompts |
| **Bulk Solid Waste Expansion** | v0.9.0 | VPMs & Pathways | Expansion to Red Mud, Steel Slag, Fly Ash, and Coal Gangue | Cross-waste valorization recipe optimization |
| **Hazardous Waste Confinement**| v1.0.0 | Hazardous VPMs | Heavy-metal leaching kinetics, NORM radioactive partition | Compliance with GB 18598 & EPA RCRA hazardous limits |

---

## 9. Strategic Evolution: From Phosphogypsum to All Bulk & Hazardous Solid Wastes

Phosphogypsum serves as the pilot proving ground for this physics-informed agentic architecture. The overarching mission of this framework is to generalize across all industrial bulk solid wastes (大宗工业固废) and hazardous wastes (危险废物):

```mermaid
graph TD
    subgraph Stage1 ["Stage 1: PhosphogypsumBot (v0.6.0 - Validated)"]
        PG["Phosphogypsum (磷石膏)"]
        PG --> VPM_PG["5 VPMs: Acid Leaching, Carbothermic, Calcination, Merseburg, REE"]
        PG --> Agent_PG["10-Tool Plan-and-Solve Agent"]
    end

    subgraph Stage2 ["Stage 2: Bulk Solid Waste Matrix (大宗固废通用化 - v0.9.0)"]
        Bulk["Bulk Industrial Wastes"]
        RM["Red Mud (赤泥 - Bayer/Sintering)"]
        SS["Steel Slag (钢渣 / 高炉矿渣)"]
        FA["Coal Fly Ash (粉煤灰 / 煤矸石)"]
        CS["Carbide Slag (电石渣)"]
        
        Bulk --> RM & SS & FA & CS
        RM & SS & FA & CS --> Unified_VPM["Generalized VPM & Multi-Waste Co-Processing Engine"]
    end

    subgraph Stage3 ["Stage 3: Hazardous Solid Waste Confinement (危废高值无害化 - v1.0.0)"]
        Haz["Hazardous Waste Stream"]
        MSWI["Incineration Fly Ash (垃圾焚烧飞灰)"]
        EP["Electroplating Sludge (电镀污泥)"]
        SM["Smelting Residues (有色冶炼废渣)"]
        
        Haz --> MSWI & EP & SM
        MSWI & EP & SM --> Haz_Solver["Multi-Phase Leaching & Radionuclide / Heavy-Metal Immobilization Engine"]
    end

    Stage1 ==> Stage2 ==> Stage3
```

### Strategic Milestones for Generalization:
1. **Universal Chemical & Phase Ontology**: Expand `pgloop/chemicals/` to cover aluminum silicate complexes (赤泥铝硅酸盐相), heavy-metal spinels (重金属尖晶石固溶相), and chloride volatilization kinetics.
2. **Co-Processing & Multi-Feedstock Blend Optimizer**: Transfer the multi-component continuous simplex optimization engine (proven in `PyroBot`) to multi-solid-waste synergistic recipes (e.g., Red Mud + Phosphogypsum + Steel Slag green geopolymer cement).
3. **Proprietary Open-Weight Foundation Model**: Release **`SolidWaste-Qwen-32B-Instruct`**, an open-source, physics-constrained large language model for international circular economy research.

