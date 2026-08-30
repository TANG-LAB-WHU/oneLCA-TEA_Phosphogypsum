# PhosphogypsumBot: Physics-Informed AI Agent Framework for Industrial Phosphogypsum Engineering

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Inference: llama.cpp](https://img.shields.io/badge/inference-llama.cpp-orange.svg)](https://github.com/ggerganov/llama.cpp)
[![Version](https://img.shields.io/badge/version-v0.6.5-green.svg)](https://github.com/TANG-LAB-WHU/oneLCA-TEA_Phosphogypsum)

**PhosphogypsumBot** is a physics-informed, multimodal intelligent agent framework designed to quantify and mitigate uncertainties in industrial phosphogypsum (PG) valorization, Life Cycle Assessment (LCA), and Techno-Economic Analysis (TEA).

By integrating physical conservation laws, Bayesian Markov Chain Monte Carlo (MCMC) sampling, high-accuracy literature extraction (IBM Docling & MinerU), and a deterministic **10-Tool Plan-and-Solve Autonomous Agent**, PhosphogypsumBot guides engineers and policy-makers toward optimal, sustainable, and economically viable circular economy pathways.

---

## Core Capabilities

### 1. Autonomous Plan-and-Solve Agent (`chat_agent/`)
* **10-Tool Portfolio**: Deterministic Python solvers exposed directly to OpenAI-compatible Function Calling interfaces (Reverse Design, Benefit Compensation, MCMC, LCA/TEA, 5D TEPES Ranking, Crystal Properties, Real-time IoT).
* **Plan-and-Solve Reasoning**: Autonomous multi-step reasoning that queries literature, calculates forward environmental-economic footprints, back-calculates process parameters, and optimizes governmental subsidy structures.
* **Unified llama.cpp Inference**: Natively designed to run with `llama-server`, enabling zero-cost, high-speed, local or supercomputing (WHU-SCC) cluster deployments.

### 2. Physics-Informed AI (PI-AI) Engine
* **Physical Governing Equations**: Valorization Pathway Modules (VPMs) located in `pgloop/pathways/vpms/` represent governing thermodynamics, chemical kinetics (e.g., shrinking core models for acid leaching, carbonation rate laws, and thermal decomposition heat balances).
* **PINNs & Density Solvers**: Resolve density evolution and transport boundaries using Physics-Informed Neural Networks (PINNs) and Fokker-Planck partial differential equation (PDE) solvers (`pgloop/stochastic_dynamics/`).
* **MCMC Parameter Calibration**: Calibrates and refines joint parameter uncertainty using Metropolis-Hastings, Hamiltonian Monte Carlo, and Gibbs sampling.

### 3. Full-Dimensional Sustainability & Reverse Design
* **Bayesian Reverse Design**: Uses `ReverseDesignOptimizer` (Gaussian Process + UCB acquisition) to back-calculate required process inputs (kiln temperature, reagent ratios, steam energy) satisfying user-defined GWP/NPV constraints.
* **Benefit Compensation Model**: Internalizes avoided environmental damage (CE Delft shadow prices) to optimize stakeholder incentive structures (tipping fees, carbon credits, governmental subsidies).
* **5D TEPES Decision Matrix**: Multi-criteria decision ranking across Technical (TRL), Economic (NPV/IRR), Environmental (ISO 14040 LCA), Policy (carbon tax/subsidies), and Social (job creation/health risk) metrics.

### 4. SOTA Ingestion & Materials Potential
* **High-Accuracy Paper Ingestion**: Supports **IBM Docling** and MinerU for high-fidelity extraction of complex multi-column tables, reaction kinetics, and chemical equations from scientific literature.
* **Materials Potential Validation**: Connects to the Materials Project API (`mp-api`) to validate static crystal energies with universal MACE machine-learning interatomic potentials, performing BFGS structural relaxations and Birch-Murnaghan Equation of State (EOS) bulk modulus fitting.
* **Industrial IoT Telemetry**: Asynchronously streams OPC UA edge sensors to an MQTT broker and persists live data into SQLite WAL databases for sub-second Streamlit monitoring.

---

## Core Modules

| Module | Description | Key Classes / Sub-modules |
| :--- | :--- | :--- |
| `chat_agent/` | Autonomous Plan-and-Solve Agent | `PhosphogypsumAgent`, `AVAILABLE_TOOLS` (10 core tools), `function_to_schema`, CLI shell |
| `pgloop/pathways` | Treatment Pathways & VPMs | `CementPathway`, `REEExtractionPathway`, `SulfurAcidPathway`, `vpms/` (Carbothermic, Crystallization, Hydration) |
| `pgloop/decision` | Multi-Criteria Decision & Optimization | `PathwayRanker`, `ReverseDesignOptimizer`, `BenefitCompensationModel`, `ScenarioAnalyzer` |
| `pgloop/uncertainty` | Uncertainty Quantification & MCMC | `MonteCarloSimulator`, `JointUncertaintyPropagator`, `MetropolisHastings`, `HamiltonianMC`, `GibbsSampler` |
| `pgloop/stochastic_dynamics` | Physics-Informed Solvers | `FP_PINN` (PINN solver), `FokkerPlanck1DSolver`, `FokkerPlanck2DSolver`, `VAE` |
| `pgloop/lca` | Life Cycle Assessment Engine | `LCAEngine`, `ImpactAssessment`, `LifeCycleInventory` |
| `pgloop/tea` | Techno-Economic Analysis Engine | `TEAEngine`, `CAPEXCalculator`, `OPEXCalculator`, `ExternalCostCalculator` |
| `pgloop/knowledge` | Knowledge Extraction & Graph | `PhosphogypsumKG`, `RAGAnythingEngine`, `LightRAGEngine`, `llm_extractor`, `embeddings/` |
| `pgloop/chemicals` | Material Database & ML Properties | `Chemical`, `PropertyPredictor`, `evaluate_mace_on_mp` (MACE validator), `optimize_structure`/`fit_eos` |
| `pgloop/equipment` | Unit Operations Modeling | `CSTRReactor`, `LeachingTank`, `MixingTank`, `SeparationFilter` |
| `pgloop/risk` | Micro & Macro Risk Assessment | `TechnicalRisk`, `OperationalRisk`, `PoliticalRisk`, `PolicyRisk`, `RiskAggregator` |
| `pgloop/visualization` | Interactive Dashboard & Reporting | `run_dashboard` (Streamlit dashboard), `ReportExporter` (Excel/HTML reports) |
| `pgloop/iodata` | Ingestion & Telemetry | `PDFParser` (Docling / MinerU / PyMuPDF), `WebScraper`, `EdgeBridge` (OPC UA -> MQTT), `StreamProcessor` |

---

## System Architecture

```text
┌─════════════════════════════════════════════════════════════════════════════┐
│                     PHOSPHOGYPSUMBOT AGENTIC ARCHITECTURE                   │
└─════════════════════════════════════════════════════════════════════════════┘

                                 User Query
                                     │
                                     ▼
       ┌───────────────────────────────────────────────────────────┐
       │     PhosphogypsumBot Plan-and-Solve Agent (chat_agent/)   │
       │           Driven by Local llama-server (Qwen-35B)         │
       └─────────────────────────────┬─────────────────────────────┘
                                     │
                  Function Calling Tool Orchestration Loop
                                     │
   ┌───────────────────┬─────────────┴───────┬───────────────────┬──────────────┐
   ▼                   ▼                     ▼                   ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌────────────────────┐ ┌──────────────┐ ┌──────────────┐
│  Docling/    │ │ Forward LCA/ │ │ Bayesian Reverse   │ │ Benefit      │ │ Bayesian     │
│  LightRAG    │ │ TEA Engine   │ │ Design (GP + UCB)  │ │ Compensation │ │ MCMC (MH)    │
│  Knowledge   │ │ ISO 14040    │ │ Target GWP / NPV   │ │ Tipping Fee  │ │ Uncertainty  │
│  Retrieval   │ │ Footprints   │ │ Parameter Inversion│ │ Subsidies    │ │ Calibration  │
└──────────────┘ └──────────────┘ └────────────────────┘ └──────────────┘ └──────────────┘
                                     │
                                     ▼
       ┌───────────────────────────────────────────────────────────┐
       │      5D TEPES (Tech, Econ, Env, Policy, Social) Report    │
       └───────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites
*   Python 3.11+
*   `llama.cpp` (precompiled binary or via local server)

### Standard Setup
```bash
# Clone the repository
git clone -b feat-phosphogypsum-bot https://github.com/TANG-LAB-WHU/oneLCA-TEA_Phosphogypsum.git
cd oneLCA-TEA_Phosphogypsum

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install the package in editable mode
pip install -e .

# Install package with optional extras (AI, Visualization, Dev tools, Docling, IoT)
pip install -e ".[ai,viz,docling,dev,iot]"
```

### HPC Deployment on Wuhan University Supercomputing Center (WHU-SCC)
WHU-SCC supports heterogeneous partitions (`a100x4`, `gpu` V100, and `9a14a` AMD EPYC CPU). PhosphogypsumBot provides a unified, self-contained Slurm script that automatically detects hardware partitions, launches `llama-server` with memory interleaving, and orchestrates the agent:

```bash
# 1. A100 GPU Partition (Recommended):
sbatch -p a100x4 --gres=gpu:1 --cpus-per-task=16 slurm_jobs/run_phosphogypsum_agent.sh

# 2. 9a14a CPU Partition (192 AMD EPYC Cores with NUMA Interleaving):
sbatch -p 9a14a --nodes=1 --cpus-per-task=192 slurm_jobs/run_phosphogypsum_agent.sh

# 3. V100 GPU Partition:
sbatch -p gpu --gres=gpu:2 --cpus-per-task=10 slurm_jobs/run_phosphogypsum_agent.sh
```

---

## Quick Start

### 1. Interactive PhosphogypsumBot Agent
Launch the autonomous AI agent in command-line interactive mode or single-query mode:

```bash
# Start interactive shell (connects to local llama-server on port 11434)
python -m chat_agent.cli

# Or run a single query directly:
python -m chat_agent.cli --query "请对比 PG-CementProd 与 PG-REEextract 的 LCA 碳足迹，并进行以渣定产政策下的最优路径推荐。"
```

Or call programmatically via Python:

```python
from chat_agent.agent import PhosphogypsumAgent

agent = PhosphogypsumAgent(
    base_url="http://127.0.0.1:11434/v1",
    model="Qwen/Qwen3.6-35B-A3B-Instruct"
)

response = agent.chat("针对磷石膏分解制酸（PG-SulfurAcid）技术，进行贝叶斯逆向参数反演，要求 GWP < 120 kg CO2-eq 且 NPV > 20 $/t。")
print(response)
```

### 2. Forward LCA-TEA Assessment
Define a pathway and run a forward calculation for 1 tonne of phosphogypsum:

```python
from pgloop.lca import LCAEngine
from pgloop.tea import TEAEngine
from pgloop.pathways import CementPathway

# Initialize computational engines
lca_engine = LCAEngine()
tea_engine = TEAEngine(country="China")

# Initialize treatment pathway
pathway = CementPathway(country="China")

# Calculate environmental and financial footprints
lca_result = lca_engine.calculate(pathway, functional_unit_value=1.0)
tea_result = tea_engine.calculate(pathway, functional_unit_value=1.0)

print(f"LCA Climate Change Impact: {lca_result.impacts['climate_change']:.2f} kg CO2-eq")
print(f"TEA Conventional Cost (CLCC): ${tea_result.clcc:.2f} per tonne")
```

### 3. Bayesian Reverse Design
Back-calculate the process parameters required to satisfy output constraints using Gaussian Process regression:

```python
from pgloop.decision.optimizer.reverse_design import ReverseDesignOptimizer

def process_evaluator(params: dict) -> dict:
    temp = params["kiln_temperature"]
    coal_ratio = params["coal_ratio"]
    gwp = 200.0 - 0.1 * temp + 1.2 * coal_ratio
    npv = -50.0 + 0.15 * temp - 0.5 * coal_ratio
    return {"gwp": gwp, "npv": npv}

parameter_bounds = {
    "kiln_temperature": (900.0, 1200.0),
    "coal_ratio": (50.0, 150.0)
}

target_constraints = {
    "gwp": {"type": "max", "value": 120.0},
    "npv": {"type": "min", "value": 20.0}
}

optimizer = ReverseDesignOptimizer(
    evaluator_fn=process_evaluator,
    parameter_bounds=parameter_bounds,
    target_constraints=target_constraints
)
result = optimizer.run(n_iterations=15, n_initial_points=5)

print("Target Constraints Satisfied:", result["constraints_satisfied"])
print("Optimal Process Parameters:", result["best_parameters"])
```

### 4. Stakeholder Benefit Compensation
Optimize subsidies and tipping fees to support a green but high-capital pathway:

```python
from pgloop.decision.benefit_compensation import BenefitCompensationModel

baseline_impacts = {"climate_change": 450.0, "acidification": 30.0}
pathway_impacts = {"climate_change": 80.0, "acidification": 5.0}

model = BenefitCompensationModel()
benefits = model.calculate_avoided_damage(baseline_impacts, pathway_impacts)
total_benefit = benefits["total_avoided_environmental_benefit"]

compensation = model.optimize_compensation(
    pathway_code="PG-ChemReco",
    clcc=75.0,
    revenue=40.0,
    avoided_environmental_benefit=total_benefit,
    target_margin=5.0
)

print("Suggested Carbon Credit (USD/t):", compensation["suggested_carbon_credit"])
print("Suggested Tipping Fee (USD/t):", compensation["suggested_tipping_fee"])
print("Suggested Government Subsidy (USD/t):", compensation["suggested_subsidy"])
```

### 5. Scientific Literature Parsing (IBM Docling)
Parse PDF papers with table structure and formula extraction into clean Markdown:

```python
from pgloop.iodata.pdf_parser import PDFParser

parser = PDFParser(parser_type="docling", output_dir="./data/raw/papers/parsed")
doc = parser.parse_pdf("./data/raw/papers/unparsed/paper_sample.pdf")

print(f"Parsed Title: {doc.title}, Total Pages: {doc.pages}")
```

---

## Testing

Run unit and integration tests across all modules:

```bash
# Run full suite (unit and deterministic tests)
pytest tests/ -v -k "not integration and not slow"

# Run agent toolchain tests specifically
pytest tests/test_agent_full_toolchain.py tests/test_chat_agent.py -v
```

---

## Contributing

We welcome contributions to PhosphogypsumBot. Please follow our workflow:
1. **Branching**: Create feature branches from `main` or `feat-phosphogypsum-bot`.
2. **Formatting & Linting**: Run `ruff check .` and `black --check .` before submitting PRs.
3. **Testing**: Ensure all 85+ unit tests pass via `pytest`.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use PhosphogypsumBot in your research, please cite:

```bibtex
@software{phosphogypsumbot_2026,
  author = {TANG Lab at Wuhan University},
  title = {PhosphogypsumBot: Physics-Informed AI Agent Framework for Industrial Phosphogypsum Engineering},
  year = {2026},
  url = {https://github.com/TANG-LAB-WHU/oneLCA-TEA_Phosphogypsum}
}
```
