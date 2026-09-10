# DataHub: Multimodal Phosphogypsum Data Hub

`datahub/` is the centralized multimodal data engineering and asset hub for **PhosphogypsumBot**.
It manages raw literature PDFs, experimental curves, life-cycle inventories, parsing intermediates, knowledge graphs, and live telemetry.

---

## 🏛️ Medallion Multimodal Architecture

```
datahub/
├── raw/                               # [Bronze Tier: Immutable Sources, Read-Only]
│   ├── papers/unparsed/               # Original literature PDFs for parsing
│   ├── experimental/                  # Experimental TG/DTG curves, SEM images, XRD profiles
│   ├── telemetry/                     # Historical plant DCS/SCADA sensor dumps
│   ├── elcd/                          # European Life Cycle Database extracts
│   ├── uslci/                         # US Life Cycle Inventory database extracts
│   ├── agribalyse/                    # French agricultural and environmental LCI
│   └── idemat/                        # Idemat eco-cost database extracts
│
├── interim/                           # [Silver Tier: Staging & Parsing Intermediates]
│   └── papers/parsed/                 # Docling / MinerU generated Markdown & extracted figures
│
├── processed/                         # [Gold Tier: Processed Assets & Persistent DBs]
│   ├── extracted_data/                # Zero-shot LLM extracted JSON entities (compositions, LCI)
│   ├── parameter_ranges/              # Statistical distributions and prior ranges
│   ├── knowledge_graph/               # Domain Knowledge Graph (NetworkX GraphML / Neo4j data)
│   ├── lightrag_db/                   # GraphRAG persistent KV and vector store
│   ├── raganything_db/                # Multimodal RAG database
│   ├── dynamic_assessment/            # Stochastic dynamics PINN & VAE checkpoints / logs
│   └── telemetry.db                   # Real-time industrial IoT sensor telemetry (SQLite-WAL)
│
├── cache/                             # API response cache and transient scrapings
│
└── templates/                         # Standardized Excel/CSV ingestion contracts
    ├── composition_template.xlsx      # Chemical composition & impurity profiles
    ├── cost_template.xlsx             # Equipment CAPEX and utility OPEX tables
    └── technology_template.xlsx       # Pathway operating conditions & constraints
```

---

## 🧭 Python Programmatic Access via `DataHub` Deep Module

Avoid hardcoding relative path strings in your scripts. Always access DataHub locations through the centralized `pgloop.iodata.DataHub` facade:

```python
from pgloop.iodata import DataHub

# Access standardized paths reliably from any working directory:
print(DataHub.RAW_PAPERS_UNPARSED)       # .../datahub/raw/papers/unparsed
print(DataHub.INTERIM_PAPERS_PARSED)     # .../datahub/interim/papers/parsed
print(DataHub.PROCESSED_LIGHTRAG_DB)     # .../datahub/processed/lightrag_db
print(DataHub.TELEMETRY_DB)              # .../datahub/processed/telemetry.db
print(DataHub.TEMPLATES_DIR)             # .../datahub/templates

# Ensure required directories exist:
DataHub.ensure_directories()
```

---

## 🔒 Version Control Governance

Large binary data files (PDFs, raw CSVs, SQLite databases, and embeddings) are strictly excluded from Git via `.gitignore`.
Only directory structures, `.gitkeep` anchors, and empty standardized templates in `templates/` are tracked.
