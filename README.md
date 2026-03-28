# PG-LCA-TEA: Phosphogypsum Life Cycle Assessment & Techno-Economic Analysis Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, AI-enhanced framework for Life Cycle Assessment (LCA) and Techno-Economic Analysis (TEA) of phosphogypsum end-of-life treatment technologies.

## Features

- **6 Treatment Pathways**: Stack disposal, cement, construction materials, soil amendment, chemical recovery, REE extraction
- **Global Coverage**: Support for all major phosphogypsum producing countries with regional context
- **AI-Enhanced Data Collection**: LLM-RAG pipeline with Knowledge Graph database
- **Multi-Criteria Decision Analysis**: TOPSIS, AHP, Pareto analysis for pathway selection
- **Risk Assessment**: Micro (technical/operational/financial) and macro (political/economic/market/policy) risk evaluation
- **Uncertainty Quantification**: Monte Carlo, MCMC, and sensitivity analysis
- **Open Source Only**: No commercial databases required
- **Modular Design**: Easy to extend with new pathways, chemicals, and equipment

## Core Modules

| Module                   | Description              | Key Classes                                                 |
| ------------------------ | ------------------------ | ----------------------------------------------------------- |
| `pgloop/lca`           | Life Cycle Assessment    | `LCAEngine`, `ImpactAssessment`, `LifeCycleInventory` |
| `pgloop/tea`           | Techno-Economic Analysis | `TEAEngine`, `CAPEXCalculator`, `OPEXCalculator`      |
| `pgloop/pathways`      | Treatment Pathways       | `CementPathway`, `REEExtractionPathway`, etc.           |
| `pgloop/chemicals`     | Chemical Database + MACE | `Chemical`, `PropertyPredictor`, `get_chemical()`     |
| `pgloop/equipment`     | Unit Operations          | `CSTR`, `FilterPress`, `Evaporator`, etc.             |
| `pgloop/risk`          | Risk Assessment          | `TechnicalRisk`, `PoliticalRisk`, `RiskAggregator`    |
| `pgloop/decision`      | Decision Support         | `PathwayRanker`, `TOPSIS`, `ScenarioAnalyzer`         |
| `pgloop/uncertainty`   | Uncertainty Analysis     | `MonteCarloSimulator`, `MetropolisHastings`             |
| `pgloop/knowledge`     | AI & Knowledge Graph     | `PhosphogypsumKG`, `LightRAGEngine`, `LLMExtractor`  |
| `pgloop/utils`         | Utilities                | `CurrencyConverter`, `UnitConverter`, `Annotation`    |
| `pgloop/visualization` | Dashboard & Reports      | `run_dashboard`, `ReportExporter`, `LCAPlots`          |
| `pgloop/iodata`        | Data Ingestion           | `PDFParser`, `WebScraper`, `APIConnector`             |

## System Architecture

```text
┌─══════════════════════════════════════════════════════════════════════════════┐
│                         PG-LCA-TEA COMPLETE ARCHITECTURE                       │
└─══════════════════════════════════════════════════════════════════════════════┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              INPUT DATA SOURCES                                 │
├───────────────┬───────────────┬───────────────┬───────────────┬─────────────────┤
│ Local Papers  │ Open Access   │ Open LCI      │ Regulatory    │ Industry        │
│               │ Literature    │ Databases     │ Databases     │ Reports         │
│               │               │               │               │                 │
│ ./data/raw/   │ • Unpaywall   │ • ELCD        │ • EPA         │ • USGS          │
│   papers/     │ • PubMed OA   │ • USLCI       │ • EU-REG      │ • IFA           │
│ • unparsed/   │ • arXiv       │ • Agribalyse  │ • UN-ECE      │ • FAO           │
│ • parsed/     │ • OpenAlex    │ • Idemat      │ • IAEA        │ • World Bank    │
└───────┬───────┴───────┬───────┴───────┬───────┴───────┬───────┴────────┬────────┘
        │               │               │               │                │
        ▼               ▼               ▼               ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐              │
│  │  PDF Parser      │  │  Web Scraper     │  │  API Connector   │              │
│  │  (MinerU/PyMuPDF)│  │  (BeautifulSoup) │  │  (REST/GraphQL)  │              │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘              │
│           └─────────────────────┼─────────────────────┘                         │
│                                 ▼                                               │
│                    ┌────────────────────────┐                                   │
│                    │  Data Standardizer     │                                   │
│                    │  • Unit conversion     │                                   │
│                    │  • Schema mapping      │                                   │
│                    │  • Quality tagging     │                                   │
│                    └───────────┬────────────┘                                   │
└────────────────────────────────┼────────────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        AI + KNOWLEDGE GRAPH LAYER                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     LLM-RAG EXTRACTION ENGINE                           │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │   │
│  │  │ Text Chunking  │─▶│ Embedding      │─▶│ Vector Store   │            │   │
│  │  │ (LangChain)    │  │ (BGE/E5)       │  │ (ChromaDB)     │            │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘            │   │
│  │                                                  │                      │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌──────▼─────────┐            │   │
│  │  │ Query Engine   │◀─│ RAG Retriever  │◀─│ LLM Extractor  │            │   │
│  │  │ (LightRAG)     │  │                │  │ (Gemini/Qwen)  │            │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                          │
│  ┌──────────────────────────────────▼──────────────────────────────────────┐   │
│  │                    KNOWLEDGE GRAPH DATABASE                             │   │
│  │                        (NetworkX / Neo4j)                               │   │
│  │                                                                         │   │
│  │    [Country]──produces──▶[PG Composition]──treated_by──▶[Technology]   │   │
│  │        │                                                    │          │   │
│  │        ▼                                              ┌─────┴─────┐    │   │
│  │    [Regulation]                                   requires  emits     │   │
│  │                                                       │      │        │   │
│  │                                                       ▼      ▼        │   │
│  │                                                  [Material][Emission] │   │
│  │                                                              │        │   │
│  │                                                              ▼        │   │
│  │                                                         [Impact]      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        ML GAP-FILLING MODULE                            │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │   │
│  │  │ Similarity     │  │ Regression     │  │ Uncertainty    │            │   │
│  │  │ Matching       │  │ Prediction     │  │ Estimation     │            │   │
│  │  │ (scikit-learn) │  │ (XGBoost)      │  │ (Monte Carlo)  │            │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          LCA-TEA CALCULATION ENGINE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         LCA MODULE                                      │   │
│  │  Functional Unit: 1 tonne phosphogypsum treated                        │   │
│  │                                                                         │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │   │
│  │  │ Goal/Scope   │──▶│ Inventory    │──▶│ Impact       │                │   │
│  │  │ Definition   │   │ Analysis     │   │ Assessment   │                │   │
│  │  └──────────────┘   └──────────────┘   └──────────────┘                │   │
│  │                                                                         │   │
│  │  Impact Categories: GWP, Acidification, Eutrophication, Human Toxicity,│   │
│  │                     Ecotoxicity, Ionizing Radiation, PM, Resources     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         TEA MODULE                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │   │
│  │  │ CLCC = CAPEX_annualized + OPEX - Product_Revenue                 │  │   │
│  │  │ SLCC = Internal_Cost(shadow) + External_Cost(emissions)          │  │   │
│  │  └──────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    UNCERTAINTY ENGINE                                   │   │
│  │  Parameter Sensitivity │ Monte Carlo Simulation │ Discernibility      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      TREATMENT PATHWAY COMPARISON                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ PG-SD   │ │ PG-CM   │ │ PG-CB   │ │ PG-SA   │ │ PG-CR   │ │ PG-RE   │       │
│  │ Stack   │ │ Cement  │ │ Constr. │ │ Soil    │ │ Chemical│ │ REE     │       │
│  │ Disposal│ │         │ │ Material│ │ Amend.  │ │ Recovery│ │ Extract │       │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │
│       └──────────┴──────────┴─────┬────┴──────────┴──────────┘                  │
│                                   ▼                                             │
│                    Comparative Analysis Matrix                                  │
│                    [Extensible: New pathways can be added]                      │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      OUTPUT & VISUALIZATION LAYER                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ Interactive     │  │ Export          │  │ Knowledge       │                 │
│  │ Dashboard       │  │ Reports         │  │ Graph Explorer  │                 │
│  │ (Streamlit)     │  │ (PDF/Excel)     │  │ (PyVis/Gephi)   │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.11+
- (Optional) Neo4j 5.0+ (for production knowledge graph)

### Setup

```bash
# Clone the repository
git clone https://github.com/ResearchGeekSQ/oneLCA-TEA_Phosphogypsum.git
cd oneLCA-TEA_Phosphogypsum

# Create and activate a virtual environment (choose one)

# Option A: venv
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# Option B: Conda
# conda create -n pgloop python=3.11
# conda activate pgloop

# Install package + optional extras (declared in pyproject.toml)
pip install -e ".[ai,viz,kg,dev]"

# Optional: advanced PDF parsing (MinerU); see optional-dependencies "pdf" in pyproject.toml
# pip install -e ".[pdf]"

# Alternative mirror of pinned deps (see requirements.txt header)
# pip install -r requirements.txt && pip install -e .
```

## Project Structure

```
oneLCA-TEA_Phosphogypsum/
├── pgloop/                          # Source code (12 modules, ~80 Python files)
│   ├── chemicals/                # Chemical database + MACE property prediction
│   ├── iodata/                   # Data ingestion layer (API, PDF, Web)
│   ├── decision/                 # MCDA, Pareto, scenario analysis
│   ├── equipment/                # Unit operations (reactors, separations)
│   ├── knowledge/                # AI & Knowledge Graph layer
│   ├── lca/                      # LCA calculation module
│   ├── pathways/                 # Treatment pathway definitions
│   ├── risk/                     # Micro & macro risk assessment
│   │   ├── micro/                # Technical, operational, financial
│   │   └── macro/                # Political, economic, market, policy
│   ├── tea/                      # TEA calculation module
│   ├── uncertainty/              # MC, MCMC, sensitivity analysis
│   ├── utils/                    # Currency, units, annotations, constants
│   └── visualization/            # Dashboard & reporting
├── config/                       # Configuration files
│   ├── settings.yaml             # Global settings
│   ├── impact_factors.yaml       # Characterization factors
│   └── unit_prices.yaml          # Chemical/material prices
├── data/                         # Data storage
│   ├── raw/                      # Raw input data
│   ├── processed/                # Processed data & KG
│   └── templates/                # Data entry templates
├── models/                       # ML models
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Unit tests
├── pyproject.toml                  # Package metadata & dependency extras
├── requirements.txt                # Flat dependency mirror (see file header)
└── README.md                       # This file
```

## Quick Start

```python
from pgloop.lca import LCAEngine
from pgloop.tea import TEAEngine
from pgloop.pathways import CementPathway
from pgloop.decision import PathwayRanker
from pgloop.risk import TechnicalRisk, RiskAggregator

# Initialize engines
lca = LCAEngine()
tea = TEAEngine()

# Define pathway
pathway = CementPathway(country="China")

# Run analysis for 1 tonne PG
lca_results = lca.calculate(pathway, functional_unit=1.0)
tea_results = tea.calculate(pathway, functional_unit=1.0)

# Risk assessment
tech_risk = TechnicalRisk().assess(trl=8, scale_factor=10)
print(f"Technical risk score: {tech_risk.score}")

# Multi-criteria ranking
ranker = PathwayRanker(lca_weight=0.3, tea_weight=0.4, risk_weight=0.3)
recommendations = ranker.rank({
    "PG-CementProd": {"npv": 25, "gwp": 80, "trl": 9},
    "PG-REEextract": {"npv": 50, "gwp": 150, "trl": 6},
})
print(f"Best pathway: {recommendations[0].pathway_name}")
```

## Treatment Pathways

| Code            | Pathway                | Description                      | TRL |
| --------------- | ---------------------- | -------------------------------- | --- |
| PG-Stack        | Stack Disposal         | Baseline: engineered stacking    | 9   |
| PG-CementProd   | Cement Production      | Use as cement retarder/additive  | 9   |
| PG-ConstructMat | Construction Materials | Bricks, plasterboard, road base  | 8-9 |
| PG-Soil         | Soil Amendment         | Direct agricultural application  | 8   |
| PG-ChemReco     | Chemical Recovery      | (NH₄)₂SO₄ + CaCO₃ production | 6-7 |
| PG-REEextract   | REE Extraction         | Rare earth element recovery      | 5-6 |

## Regional Contexts

Built-in support for regional scenario analysis:

| Region                | Key Parameters                                     |
| --------------------- | -------------------------------------------------- |
| China (Yunnan)        | Low-carbon grid (hydro 70%), low labor cost        |
| Morocco (Jorf Lasfar) | Massive PG production (30 Mt/yr), high REE content |
| USA (Florida)         | Strict environmental regulations                   |
| Brazil (Minas Gerais) | Ultra-low carbon grid (hydro 65%)                  |

## Risk Assessment Framework

### Micro Risks (Project-level)

- **Technical**: TRL, scale-up, complexity
- **Operational**: Capacity, feedstock, quality
- **Financial**: Leverage, IRR, payback

### Macro Risks (Country-level)

- **Political**: Stability, regulation, corruption
- **Economic**: Monetary policy, credit, FX volatility
- **Market**: Price volatility, demand, competition
- **Policy**: Subsidies, carbon price, permits

## Data Sources (Open Access Only)

- **ELCD**: European Reference Life Cycle Database
- **USLCI**: US Life Cycle Inventory Database
- **Agribalyse**: French agricultural LCI
- **EPA**: US Environmental Protection Agency
- **IAEA**: International Atomic Energy Agency (radiation data)

## Contributing

Pull requests are welcome. MIT license applies to contributions the same as to the rest of the repository.

**Branching:** Fork and base your work on **`main`**. That branch is the default for contributors. The **`dev`** branch is reserved for maintainer-side iteration and is not the recommended starting point for forks or external PRs.

1. **Fork** the repository (use **`main`** as the default branch when cloning your fork) and create a branch for your changes.
2. **Install** with dev tools: `pip install -e ".[dev]"` (or include `ai,viz,kg` if your change touches those areas).
3. **Run checks** before opening a PR:
   - `pytest` — unit tests under `tests/`
   - `ruff check .` and `black --check .` — lint and format (line length 100 per `pyproject.toml`)
4. **Open a pull request** targeting **`main`** with a short description of the change and, if relevant, how you tested it.

For larger features or API changes, opening an issue first helps align on design and avoids duplicate work.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{pg_lca_tea_2026,
  title = {PG-LCA-TEA: Phosphogypsum Life Cycle Assessment and Techno-Economic Analysis Framework},
  year = {2026},
  url = {https://github.com/ResearchGeekSQ/oneLCA-TEA_Phosphogypsum}
}
```

## References

1. ISO 14040:2006. Environmental management — Life cycle assessment — Principles and framework.
2. ISO 14044:2006. Environmental management — Life cycle assessment — Requirements and guidelines.
