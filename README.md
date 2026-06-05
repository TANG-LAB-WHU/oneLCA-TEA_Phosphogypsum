# PhosphogypsumBot: Physics-Informed AI Agent Framework for Industrial Phosphogypsum Engineering

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-v0.5.0-green.svg)](https://github.com/TANG-LAB-WHU/oneLCA-TEA_Phosphogypsum)

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

### 5. Microscopic Materials Potential & Live IoT Telemetry (v0.5.0)
*   **Materials Potential Validation**: Connects to the Materials Project API (`mp-api`) to retrieve real DFT crystal structures (anhydrite, gypsum, fluorite) and validates static potential energy predictions with MACE machine-learning potentials.
*   **Lattice Optimizer & EOS Fitting**: Performs BFGS/FIRE structural relaxations (positions, cell shapes, volume) and Birch-Murnaghan Equation of State (EOS) fitting to determine bulk modulus properties.
*   **Edge-to-Cloud IoT Ingestion**: Streamlines asynchronous industrial sensor telemetry from edge OPC UA nodes (`asyncua`) and publishes to MQTT brokers with QoS 1, connection resilience, and LWT.
*   **Non-blocking Live Monitoring**: Persists telemetry in a SQLite database in Write-Ahead Logging (WAL) mode for concurrency, rendering instant LCA/TEA metrics inside Streamlit using `@st.fragment` sub-second updates.

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
| `pgloop/chemicals` | Material Database & ML Properties | `Chemical`, `PropertyPredictor`, `evaluate_mace_on_mp` (MACE validator), `optimize_structure`/`fit_eos` (BFGS / Birch-Murnaghan lattice optimizer) |
| `pgloop/equipment` | Unit Operations Modeling | `CSTRReactor`, `LeachingTank`, `MixingTank`, `SeparationFilter` |
| `pgloop/risk` | Micro & Macro Risk Assessment | `TechnicalRisk`, `OperationalRisk`, `PoliticalRisk`, `PolicyRisk`, `RiskAggregator` |
| `pgloop/simulation` | Multi-Scale System Simulation | `micro/` (reaction level), `meso/` (plant level), `macro/` (market/grid level) |
| `pgloop/visualization` | Interactive Dashboard & Reporting | `run_dashboard` (Streamlit dashboard), `ReportExporter` (Excel/HTML reports) |
| `pgloop/iodata` | Ingestion & Telemetry | `PDFParser` (MinerU/PyMuPDF parser), `WebScraper`, `DataStandardizer`, `EdgeBridge` (OPC UA -> MQTT), `StreamProcessor` (MQTT -> SQLite WAL) |

---

## System Architecture

```text
┌─════════════════════════════════════════════════════════════════════════════┐
│                       PHOSPHOGYPSUMBOT AGENT FRAMEWORK                      │
└─════════════════════════════════════════════════════════════════════════════┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      1. MULTIMODAL RAG DATA FOUNDATION                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Unstructured PDFs (MinerU) ────▶ Vector Database (Local Embeddings)       │
│  Knowledge Graph (Neo4j) ◄──────▶ LLM Information Extractor                 │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ (Literature facts & priors)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    2. PHYSICS-INFORMED AI (PI-AI) ENGINE                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Governing Equations: ODEs/PDEs kinetics & energy-mass conservation         │
│  Solvers: PINNs (PyTorch) + 1D/2D Fokker-Planck Density Propagators         │
│  Bayesian MCMC: Metropolis-Hastings / Hamiltonian MC / Gibbs sampling       │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ (Physics constraints & parameters)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   3. MULTI-SCALE MULTI-OBJECTIVE SYSTEM                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Thermodynamic Inputs: Heat [Thermal] + Work [Electric]                     │
│  Economic Inputs: Currency [CAPEX/OPEX/Revenues]                            │
│  Scales: Micro (Kinetics) ──▶ Meso (Plant LCA-TEA) ──▶ Macro (Grid/Mkt)     │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ (Process outputs & economics)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      4. FULL-DIMENSIONAL AI ASSESSMENT                      │
├──────────────────────────────┬──────────────────────────┬───────────────────┤
│    UNCERTAINTY QUANTIFICATION│  BAYESIAN REVERSE DESIGN │   BENEFIT COMP.   │
├──────────────────────────────┼──────────────────────────┼───────────────────┤
│ • Monte Carlo Propagation    │ • Gaussian Process (GP)  │ • Shadow Price    │
│ • Sobol Sensitivity analysis │ • Acquisition (UCB)      │   valuation       │
│ • Discernibility Analysis    │ • Parameter Target search│ • Subsidy Opt.    │
└──────────────────────────────┴─────────────┬────────────┴───────────────────┘
                                             │ (Optimization recommendations)
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     5. 5D SUSTAINABLE DEVELOPMENT LOOP                      │
├────────────┬────────────┬─────────────────┬─────────────────┬───────────────┤
│ Technical  │  Economic  │  Environmental  │     Policy      │    Social     │
│ TRL/Scale  │ NPV / IRR  │  ISO 14040 LCA  │   Carbon Tax    │ Job Creation  │
│Feasibility │  Payback   │  GWP/Toxicity   │    Subsidies    │  Health Risk  │
└────────────┴────────────┴─────────────────┴─────────────────┴───────────────┘
```

---

## Installation

### Prerequisites
*   Python 3.11+
*   (Optional) Neo4j 5.0+ (for production-grade Knowledge Graph)

### Standard Setup
```bash
# Clone the repository
git clone -b main https://github.com/TANG-LAB-WHU/oneLCA-TEA_Phosphogypsum.git
cd oneLCA-TEA_Phosphogypsum

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install the package in editable mode
pip install -e .

# Install package with optional extras (AI, Visualization, Knowledge Graph, Dev tools, RAG, IoT)
pip install -e ".[ai,viz,kg,dev,rag,iot]"
```

### HPC Deployment & vLLM Compilation (Wuhan University Supercomputing Center - WHU-SCC)
When deploying on high-performance computing (HPC) nodes like **WHU-SCC**, installing the optional packages triggers a compilation of `vllm` (required by the MinerU multi-modal parser). Since `vllm` tries to compile from source in the absence of a pre-built wheel, it requires standard CUDA compiler toolchains and build dependencies.

Follow this sequential setup inside your active virtual environment (e.g., `conda activate pgbot`) on the cluster terminal:

```bash
# 1. Load the CUDA module matching PyTorch (CUDA 12.9 is recommended)
module load nvidia/cuda/12.9

# 2. Set the CUDA_HOME environment variable (critical for the vllm compilation script)
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo "Current CUDA_HOME is: $CUDA_HOME"

# 3. Pre-install fundamental build-time dependencies
pip install numpy ninja wheel setuptools

# 4. Install the package in editable mode with GPU acceleration indices
pip install -e ".[ai,viz,rag,kg,dev,iot]" --extra-index-url https://download.pytorch.org/whl/cu129
```

For detailed Slurm job scripts, compute node partition guidelines (`a100x4`, `gpu`, `9a14a`), and NUMA socket isolation (`numactl`) details, refer to the **[HPC Supercomputer Deployment Guide](docs/hpc_slurm_deployment.md)**.

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

### 5. Microscopic Materials Potential Evaluation (MACE & MP-API)
Fetch real DFT crystal structures from the Materials Project API online database and validate the accuracy of zero-shot energy predictions using MACE universal machine-learning interatomic potentials:

```python
import os
from pgloop.chemicals.eval_mace import evaluate_mace_on_mp

# MP IDs representing Phosphogypsum phases and co-existing impurities:
# 1. CaSO4 (Anhydrite): mp-4406
# 2. CaSO4.2H2O (Gypsum): mp-23690
# 3. CaF2 (Fluorite): mp-2741
default_ids = ["mp-4406", "mp-23690", "mp-2741"]

# Evaluate MACE-MP static potential energy against DFT benchmarks
# Note: Requires setting the 'MP_API_KEY' environment variable.
results = evaluate_mace_on_mp(default_ids, model_size="medium", device="cpu")

print("Evaluation Success (MAE < 0.05 eV/atom):", results["success"])
print(f"Mean Absolute Error: {results['mae']:.6f} eV/atom")
```

Perform structure geometry/cell relaxations and Equation of State (EOS) fitting:

```python
import os
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from pgloop.chemicals.mace_interface import get_mace_calculator
from pgloop.chemicals.lattice_optimizer import optimize_structure, fit_eos

# 1. Fetch structure from Materials Project
with MPRester(os.environ.get("MP_API_KEY")) as mpr:
    struct = mpr.summary.get_data_by_id("mp-4406").structure
atoms = AseAtomsAdaptor.get_atoms(struct)

# 2. Setup MACE machine-learning calculator
calc = get_mace_calculator(model_size="medium")

# 3. Relax atomic coordinates, cell shapes, and volume simultaneously
relaxed_atoms, metadata = optimize_structure(
    atoms=atoms,
    calculator=calc,
    fmax=0.05,
    constant_volume=False
)
print(f"Optimized Energy: {metadata['energy_per_atom_ev']:.4f} eV/atom")

# 4. Fit Birch-Murnaghan Equation of State (EOS) to get Bulk Modulus
eos, eos_results = fit_eos(
    atoms=relaxed_atoms,
    calculator=calc,
    num_points=7,
    strain_range=0.05
)
print(f"Fitted Bulk Modulus: {eos_results['b0_gpa']:.2f} GPa")
```

### 6. Industrial IoT Stream Ingestion & Live Monitoring
PhosphogypsumBot supports a production-grade, real-time edge telemetry pipeline. The data flow runs from physical sensors (OPC UA) -> Edge Bridge -> MQTT Broker -> Stream Processor (LCA/TEA validation) -> SQLite WAL Database -> Streamlit Live Dashboard.

#### Step 1: Start the Edge Bridge (OPC UA -> MQTT)
Subscribe to target industrial OPC UA nodes and forward them to an MQTT broker. Implements QoS 1, connection resilience, and LWT status retention:

```bash
# Start the Edge Bridge (runs asynchronously)
# Provide the OPC UA Node IDs to subscribe to as arguments
python pgloop/iodata/edge_bridge.py "ns=2;i=2" "ns=2;i=3"
```

#### Step 2: Start the Stream Processor (MQTT -> SQLite WAL)
Subscribe to the raw MQTT telemetry stream, perform physics conservation rules and boundary validation checks, compute instant LCA (CO2 emission rate) and TEA (OPEX cost rate) KPIs, and persist records into a SQLite database with Write-Ahead Logging (WAL) mode enabled:

```bash
# Start the Stream Processor
python pgloop/iodata/stream_processor.py
```

#### Step 3: Launch the Dashboard
Run the Streamlit interactive dashboard and navigate to the **Live Monitoring** tab to view real-time metrics, auto-refreshing every second using non-blocking `@st.fragment` renders:

```bash
# Start the dashboard
streamlit run pgloop/visualization/dashboard.py
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
