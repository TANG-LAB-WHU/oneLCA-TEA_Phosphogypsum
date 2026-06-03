# PhosphogypsumBot: Physics-Informed AI Agent Framework for Industrial Phosphogypsum Engineering

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PhosphogypsumBot** is a physics-informed, multimodal intelligent agent framework designed to quantify and mitigate uncertainties in industrial phosphogypsum (PG) valorization, life cycle assessment (LCA), and techno-economic analysis (TEA).

By integrating physical conservation laws, Bayesian Markov Chain Monte Carlo (MCMC) sampling, and Multimodal Retrieval-Augmented Generation (RAG), PhosphogypsumBot guides engineers and policy-makers toward optimal, sustainable, and economically viable circular economy pathways.

---

## Core Capabilities

### 1. Physics-Informed AI (PI-AI) Engine
*   **Physical Governing Equations**: Valorization Pathway Modules (VPMs) located in `pgloop/pathways/vpms/` represent governing thermodynamics, chemical kinetics (e.g., shrinking core models for acid leaching, carbonation rate laws, and thermal decomposition heat balances).
*   **PINNs & Density Solvers**: Resolve density evolution and transport boundaries using Physics-Informed Neural Networks (PINNs) and Fokker-Planck partial differential equation (PDE) solvers (`pgloop/stochastic_dynamics/`).
*   **MCMC Parameter Calibration**: Calibrates and refines joint parameter uncertainty using Metropolis-Hastings, Hamiltonian Monte Carlo, and Gibbs sampling.

### 2. Multi-Scale & Multi-Objective Process Modeling
*   Tracks thermodynamic inputs of heat (**Heat** - e.g., coal, natural gas, steam) and electricity (**Work** - e.g., agitation, filtration, pumping) and evaluates them alongside cost inputs (**Currency** - e.g., CAPEX, raw materials, labor).
*   Models processes from micro-scale (chemical reactions in unit operations) to meso-scale (process pathway scaling) and macro-scale (regional electricity grid mixes, carbon tax structures).

### 3. Full-Dimensional Sustainability Assessment
*   **Uncertainty Quantification**: Propagates process variance via Monte Carlo simulations, Sobol sensitivity screening, and pathway discernibility assessments.
*   **Bayesian Reverse Design**: Uses the `ReverseDesignOptimizer` to back-calculate input process variables (e.g., kiln temperature, raw material purity, or grid mix) required to satisfy target environmental and economic thresholds (e.g., GWP <= 100 kg CO2-eq/t, NPV >= $20/t).
*   **System Benefit Compensation**: Leverages the `BenefitCompensationModel` to internalize environmental external benefits (monetized using shadow pricing) and optimize stakeholder incentive structures, including tipping fees, carbon credits, and governmental subsidies.

### 4. 5D Sustainable Development Loop
Integrates multi-criteria decision analysis (TOPSIS/AHP) across five core pillars:
1.  **Technical**: Technology Readiness Level (TRL) and scale-up feasibility.
2.  **Economic**: Financial returns (NPV, IRR, payback period).
3.  **Environmental**: ISO 14040/14044 LCA (global warming, human toxicity, resource depletion).
4.  **Policy**: Geopolitical risks, carbon pricing, and regional subsidy dependencies.
5.  **Social**: Employment creation (`job_creation`) and community health/safety risks (`community_health_risk`).

---

## Core Modules

| Module | Description | Key Classes / Sub-modules |
| :--- | :--- | :--- |
| `pgloop/pathways` | Treatment Pathways & VPMs | `CementPathway`, `REEExtractionPathway`, `SulfurAcidPathway`, `vpms/` (Carbothermic, Crystallization, Hydration) |
| `pgloop/decision` | Multi-Criteria Decision & Optimization | `PathwayRanker`, `ReverseDesignOptimizer`, `BenefitCompensationModel`, `ScenarioAnalyzer` |
| `pgloop/uncertainty` | Uncertainty Quantification & MCMC | `MonteCarloSimulator`, `JointUncertaintyPropagator`, `MetropolisHastings`, `HamiltonianMC`, `GibbsSampler` |
| `pgloop/stochastic_dynamics` | Physics-Informed Solvers | `FP_PINN` (PINN solver), `FokkerPlanck1DSolver`, `FokkerPlanck2DSolver`, `VAE` |
| `pgloop/lca` | Life Cycle Assessment Engine | `LCAEngine`, `ImpactAssessment`, `LifeCycleInventory` |
| `pgloop/tea` | Techno-Economic Analysis Engine | `TEAEngine`, `CAPEXCalculator`, `OPEXCalculator`, `ExternalCostCalculator` |
| `pgloop/knowledge` | Knowledge Extraction & Graph | `PhosphogypsumKG`, `RAGAnythingEngine`, `LightRAGEngine`, `embeddings/` |
| `pgloop/chemicals` | Material Database & ML Properties | `Chemical`, `PropertyPredictor` (MACE machine-learning property predictor) |
| `pgloop/equipment` | Unit Operations Modeling | `CSTRReactor`, `LeachingTank`, `MixingTank`, `SeparationFilter` |
| `pgloop/risk` | Micro & Macro Risk Assessment | `TechnicalRisk`, `OperationalRisk`, `PoliticalRisk`, `PolicyRisk`, `RiskAggregator` |
| `pgloop/simulation` | Multi-Scale System Simulation | `micro/` (reaction level), `meso/` (plant level), `macro/` (market/grid level) |
| `pgloop/visualization` | Interactive Dashboard & Reporting | `run_dashboard` (Streamlit dashboard), `ReportExporter` (Excel/HTML reports) |
| `pgloop/iodata` | Raw Data Ingestion | `PDFParser` (MinerU/PyMuPDF parser), `WebScraper`, `DataStandardizer` |

---

## System Architecture

```text
┌─══════════════════════════════════════════════════════════════════════════════┐
│                         PHOSPHOGYPSUMBOT AGENT FRAMEWORK                       │
└─══════════════════════════════════════════════════════════════════════════════┘

   ┌─────────────────────────────────────────────────────────────────────────┐
   │                          MULTIMODAL RAG INGESTION                       │
   │  • Raw literature (PDFs/MinerU) ──▶ Vector Store (Ollama Embedding API) │
   │  • LLM Extractor ────────────────▶ Knowledge Graph (Neo4j / NetworkX)   │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                       PHYSICS-INFORMED AI (PI-AI)                       │
   │  • Governing Equations (VPMs: Carbothermic, Hydration, Crystallization)  │
   │  • Transport & Density evolution solved via PINNs / Fokker-Planck PDEs  │
   │  • Bayesian MCMC Calibration (Metropolis-Hastings / Hamiltonian MC)    │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        │ (Injects physical priors & params)
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                     MULTI-SCALE MULTI-OBJECTIVE SYSTEM                  │
   │                Input Flows: Heat [热] + Work [功] + Currency [货币]     │
   ├─────────────────────────────────────────────────────────────────────────┤
   │  [Micro: Reaction Kinetics] ──▶ [Meso: Unit Op / LCA-TEA] ──▶ [Macro]    │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                         FULL-DIMENSIONAL ASSESSMENT                     │
   ├──────────────────────────────┬──────────────────────────┬───────────────┤
   │     UNCERTAINTY INFERENCE    │   REVERSE DESIGN OPT.    │  BENEFIT COMP.│
   │   • Monte Carlo Propagation  │ • Bayesian Optimization  │• Shadow Price │
   │   • Elasticity Sensitivity   │   (Gaussian Process GP)  │  Internalization│
   │   • Discernibility Analysis  │ • Parameter Target search│• Subsidy Opt. │
   └──────────────────────────────┴─────────────┬────────────┴───────────────┘
                                                │
                                                ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                      5D SUSTAINABLE DEVELOPMENT LOOP                    │
   │   Technical  ◀──▶  Economic  ◀──▶  Environmental  ◀──▶  Policy  ◀──▶ Social │
   │   (TRL/Scale)      (NPV/IRR)       (ISO LCA GWP)      (Subsidies) (Jobs/Risk)│
   └─────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites
*   Python 3.11+
*   (Optional) Neo4j 5.0+ (for production-grade Knowledge Graph)

### Setup
```bash
# Clone the repository
git clone -b main https://github.com/TANG-LAB-WHU/oneLCA-TEA_Phosphogypsum.git
cd oneLCA-TEA_Phosphogypsum

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install the package in editable mode
pip install -e .

# Install package with optional extras (AI, Visualization, Knowledge Graph, Dev tools, RAG)
pip install -e ".[ai,viz,kg,dev,rag]"
```

---

## Quick Start

### 1. Forward LCA-TEA Assessment
Define a pathway and run a forward calculation for 1 tonne of phosphogypsum:

```python
from pgloop.lca import LCAEngine
from pgloop.tea import TEAEngine
from pgloop.pathways import CementPathway

# Initialize computational engines
lca_engine = LCAEngine()
tea_engine = TEAEngine(country="China")

# Initialize treatment pathway (uses natural gas/coal heat, electric work, regional params)
pathway = CementPathway(country="China")

# Calculate environmental and financial footprints
lca_result = lca_engine.calculate(pathway, functional_unit_value=1.0)
tea_result = tea_engine.calculate(pathway, functional_unit_value=1.0)

print(f"LCA Climate Change Impact: {lca_result.impacts['climate_change']:.2f} kg CO2-eq")
print(f"TEA Conventional Cost (CLCC): ${tea_result.clcc:.2f} per tonne")
```

### 2. Bayesian Reverse Design
Back-calculate the process parameters required to satisfy output constraints (e.g., GWP <= 120 kg CO2-eq, NPV >= $20/t) using Bayesian Optimization:

```python
from pgloop.decision import ReverseDesignOptimizer

# Define forward evaluator function mapping inputs to outputs
def process_evaluator(params: dict) -> dict:
    temp = params["kiln_temperature"]
    coal_ratio = params["coal_ratio"]
    
    # Forward simulation surrogate
    gwp = 200.0 - 0.1 * temp + 1.2 * coal_ratio
    npv = -50.0 + 0.15 * temp - 0.5 * coal_ratio
    return {"gwp": gwp, "npv": npv}

# Parameter boundaries for search space
parameter_bounds = {
    "kiln_temperature": (900.0, 1200.0),
    "coal_ratio": (50.0, 150.0)
}

# Constraint targets (GWP must be minimized, NPV maximized)
target_constraints = {
    "gwp": {"type": "max", "value": 120.0},
    "npv": {"type": "min", "value": 20.0}
}

# Run Bayesian optimization
optimizer = ReverseDesignOptimizer(
    evaluator_fn=process_evaluator,
    parameter_bounds=parameter_bounds,
    target_constraints=target_constraints
)
result = optimizer.run(n_iterations=15, n_initial_points=5)

print("Target Constraints Satisfied:", result["constraints_satisfied"])
print("Optimal Process Parameters:", result["best_parameters"])
print("Surrogate Sensitivities:", result["parameter_sensitivities"])
```

### 3. Stakeholder Benefit Compensation
Optimize subsidies and tipping fees to support a green but high-capital pathway:

```python
from pgloop.decision import BenefitCompensationModel

# Define baseline stack disposal vs advanced chemical recovery impacts
baseline_impacts = {"climate_change": 450.0, "acidification": 30.0}
pathway_impacts = {"climate_change": 80.0, "acidification": 5.0}

model = BenefitCompensationModel()

# 1. Monetize avoided environmental damages
benefits = model.calculate_avoided_damage(baseline_impacts, pathway_impacts)
total_benefit = benefits["total_avoided_environmental_benefit"]

# 2. Optimize compensation structure for a pathway with high CLCC ($75/t) and low revenue ($40/t)
compensation = model.optimize_compensation(
    pathway_code="PG-ChemReco",
    clcc=75.0,
    revenue=40.0,
    avoided_environmental_benefit=total_benefit,
    target_margin=5.0
)

print("Compensation Status:", compensation["status"])
print("Suggested Carbon Credit (USD/t):", compensation["suggested_carbon_credit"])
print("Suggested Tipping Fee (USD/t):", compensation["suggested_tipping_fee"])
print("Suggested Government Subsidy (USD/t):", compensation["suggested_subsidy"])
```

### 4. MCMC Uncertainty Calibration
Run MCMC sampling to calibrate correlated parameters:

```python
import numpy as np
from pgloop.uncertainty.mcmc import MetropolisHastings

# Define prior log probability function
def log_prior(theta):
    if 0 < theta[0] < 10 and 0 < theta[1] < 10:
        return 0.0
    return -np.inf

# Define log likelihood
def log_likelihood(theta):
    # Simulated comparison against experimental measurements
    x, y = theta
    return -0.5 * ((x - 2.5) ** 2 + (y - 3.8) ** 2)

def log_prob(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# Initialize MCMC
sampler = MetropolisHastings(
    log_prob_fn=log_prob,
    parameter_names=["x", "y"],
    initial_state=np.array([1.0, 1.0])
)
mcmc_result = sampler.sample(n_samples=2000, warmup=500)
summary = mcmc_result.summary()

print("Posterior Means:", summary["means"])
print("95% Credible Intervals:", summary["95_credible"])
```

---

## Contributing

We welcome contributions to PhosphogypsumBot. Please read our guidelines:

1.  **Branching**: Base all changes on the `main` branch. The `dev` branch is reserved for maintainer-side release integration.
2.  **Formatting & Linting**: Run `ruff check .` and `black --check .` before submitting pull requests (max line length: 100).
3.  **Testing**: Verify code integrity with `pytest` inside the root directory.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use PhosphogypsumBot in your research, please cite:

```bibtex
@software{phosphogypsumbot_2026,
  title = {PhosphogypsumBot: Physics-Informed AI Agent Framework for Industrial Phosphogypsum Engineering},
  year = {2026},
  url = {https://github.com/TANG-LAB-WHU/oneLCA-TEA_Phosphogypsum}
}
```
