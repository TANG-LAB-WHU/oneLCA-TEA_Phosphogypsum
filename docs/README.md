# PhosphogypsumBot Technical Documentation Hub

Welcome to the central engineering documentation hub for the **PhosphogypsumBot** framework. While the [root README](../README.md) provides a high-level overview of the project's vision, core capabilities, and usage examples, this `docs/` directory is dedicated to **in-depth deployment architectures, implementation internals, and mathematical methodology references**.

This entire platform, including its heavy multi-modal Knowledge Graph parsing, GraphRAG indexing, and agent tool-calling execution pipelines, has been custom-tailored, deployed, and rigorously validated on the **Wuhan University Supercomputing Center (WHU-SCC, 武汉大学超级计算中心)**.

---

## 📖 Documentation Index

### 1. High-Performance Computing (HPC) & SLURM Deployment
Detailed guide on deploying the compute-heavy, dual-LLM pipelines on supercomputer clusters (specifically optimized for WHU-SCC partitions) using `llama.cpp` server backends and NUMA-node process pinning.
*   **[HPC Supercomputer Deployment Guide](hpc_slurm_deployment.md)**
    *   Conda environment configuration and environment variables.
    *   NUMA node isolation (`numactl --cpunodebind`) to resolve memory-bandwidth bottlenecks.
    *   SLURM job script blueprints (Scenario A: A100 Dual-GPU, Scenario B: V100 Multi-GPU, Scenario C: EPYC CPU-only partitions).
    *   Interactive srun sessions for terminal-based agent chat dialogs.

### 2. Knowledge Graph (KG) Ingestion & Extraction Pipeline
The step-by-step engineering pipeline that digests unstructured scientific papers (PDFs) into structured facts and statistical ranges.
*   **[Multimodal KG Extraction Architecture](knowledge_graph_pipeline.md)**
    *   **Phase 1: PDF Layout Parsing**: Extraction of text, complex formulas, and charts using MinerU and PyMuPDF, utilizing a 35B Vision model.
    *   **Phase 2: GraphRAG Ingest**: LightRAG-based vector and relational graph building.
    *   **Phase 3: LLM Information Extractor**: Structured JSON extraction for chemical compositions, LCI metrics, and policy parameters.
    *   **Phase 4 & 5: Ranges & Graph Construction**: Compiling parameter ranges and constructing the final knowledge graph.

### 3. Agentic Framework & RAG Integration
Documentation on the Plan-and-Solve CoT orchestration engine that connects the user interface to physics solvers and databases.
*   **[Chat Agent Internal Mechanisms](chat_agent_internals.md)**
    *   Autonomous tool-calling loop in `chat_agent/agent.py` using OpenAI-compatible function calling schemas.
    *   Full 10-Tool portfolio: `calculate_lca_tea`, `rank_all_pathways`, `run_market_robustness_scenario`, `get_available_pathways`, `search_literature`, `optimize_reverse_design`, `optimize_benefit_compensation`, `calibrate_process_parameters`, `predict_crystal_properties`, `query_realtime_telemetry`.
    *   System Prompt safety constraints, Plan-and-Solve 5-step decomposition, and formatting guidelines.

### 4. Sustainability & Process Methodologies
Rigorous mathematical, physical, and economic modeling underpinning the pathway simulations.
*   **[Life Cycle Assessment (LCA) Framework](methodology_lca.md)**
    *   ISO 14040/14044 compliance, 10 environmental impact categories, and inventory scaling.
    *   Monte Carlo uncertainty propagation from parameter distribution inputs.
*   **[Techno-Economic Analysis (TEA) Modeling](methodology_tea.md)**
    *   Conventional Life Cycle Costing (CLCC) and Societal Life Cycle Costing (SLCC) with shadow pricing.
    *   CAPEX, OPEX, and revenue breakdowns.

### 5. Physics-Informed VPM (Valorization Pathway Module) Systems
> **VPM Definition**: **VPM** stands for **Valorization Pathway Module**. It serves as the standardized, physics-constrained software abstraction for individual industrial phosphogypsum valorization routes. Each VPM strictly defines governing reaction kinetics, mass/energy conservation equations, input energy vectors $U_{in}$ (Heat, Work, Money), and NORM radionuclide partition boundaries.

Detailed explanation of physical kinetics, thermodynamics, and mass transfer governing equations for each chemical pathway:
*   **[B1 VPM_carbothermic_reduction: Carbothermic Reduction](vpm_b1_carbothermic_reduction.md)**: Thermochemical reduction kinetics of PG to recover sulfuric acid and cement clinker.
*   **[B2 VPM_hydration: Hydration & Ettringite](vpm_b2_hydration_ettringite.md)**: Curing hydration kinetics, crystallization-pressure, and structural solidification parameters.
*   **[B3 VPM_ammono_carbonation: Ammonium Carbonation](vpm_b3_ammono_carbonation.md)**: Three-phase Merseburg process kinetics, gas dissolution, and carbon mineral sequestration.
*   **[B4 VPM_alpha_hemihydrate: α-Hemihydrate Calcination](vpm_b4_alpha_hemihydrate.md)**: Hydrothermal phase crystallization kinetics and crystal aspect ratio modifiers.
*   **[B5 VPM_ree_extraction: Rare Earth Element Acid Leaching](vpm_b5_ree_extraction.md)**: Shrinking core diffusion model for selective acid leaching of REEs from PG.

---

## 🛠️ Prerequisites & Environment Configuration

To run the pipelines and agent locally or on a compute node, you must set up the project environment variables. Create a `.env` file in the project root:

```ini
# Core LLM API Config (Reasoner LLM - Port 11434)
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_API_KEY=sk-no-key-required
LLM_MODEL=qwen2.5:32b                  # E.g., Qwen3.6-27B-Instruct or Qwen2.5:32b
LLM_CONTEXT_LENGTH=32768
LLM_JSON_MODE=1

# Embedding API Config (Embedding Model - Port 11436 or 11434)
EMBEDDING_BASE_URL=http://127.0.0.1:11436/v1
EMBEDDING_API_KEY=sk-no-key-required
EMBEDDING_MODEL=qwen3-embedding:4b     # E.g., Qwen3-Embedding-8B-Q8_0

# (Optional) Parser VLM Config (Parser VLM - Port 11435)
PARSER_VLM_URL=http://127.0.0.1:11435/v1
PARSER_VLM_MODEL=qwen3.6-35b-vision

# Performance & Timeouts
LLM_TIMEOUT=3600                       # Seconds (useful for slow CPU inferences)
EMBEDDING_TIMEOUT=1200
TIKTOKEN_CACHE_DIR=/home/tangsiqi/.cache/tiktoken  # Critical for offline supercomputer nodes
```

---

## 📁 Data Directory Hierarchy

The extraction pipeline reads from and writes to the structured `data/` folder hierarchy. Developers must follow this directory structure:

```text
data/
├── raw/
│   └── papers/
│       ├── unparsed/          <-- Put raw scientific PDF papers here
│       └── parsed/            <-- Markdown outputs generated by PDFParser (MinerU/PyMuPDF)
└── processed/
    ├── lightrag_db/           <-- GraphRAG database files (relational graph + vector indices)
    ├── raganything_db/        <-- Multimodal vector indexes (embeddings of images and charts)
    ├── extracted_data/        <-- Structured JSON files (*_extracted.json) generated by LLM Extractor
    ├── parameter_ranges/      <-- Extracted statistical parameter ranges (*_ranges.json)
    └── knowledge_graph/       <-- The final compiled domain knowledge graph (Neo4j / NetworkX formats)
```

---

## 🚀 Quick Reference: SLURM Pipeline Execution

For developers running the pipeline on the **Wuhan University Supercomputing Platform**, execute the following command sequences:

**1. Drop raw papers** (PDF format) into [data/raw/papers/unparsed/](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/data/raw/papers/unparsed/).

**2. Submit the Slurm Job** based on partition resources:
*   **CPU Partition (9a14a - 192 cores)**:
    ```bash
    sbatch slurm_jobs/test_kg_pipeline_cpu.sh
    ```
*   **NVIDIA A100 GPU Partition**:
    ```bash
    sbatch slurm_jobs/test_kg_pipeline_a100.sh
    ```
*   **NVIDIA V100 GPU Partition**:
    ```bash
    sbatch slurm_jobs/test_kg_pipeline_v100.sh
    ```

**3. Monitor logs** in real-time:
```bash
tail -f slurm_jobs/logs/test_kg_pipeline_cpu/kg_test_cpu_<job_id>.log
```

> **⚠️ Critical Architecture Note**: 
> The CPU scripts use `numactl --cpunodebind=0` and `numactl --cpunodebind=1` to pin the LLM and embedding processes to distinct sockets. **Do not modify these thread allocations or CPU affinities**. They prevent severe cross-socket NUMA bus latency and memory bandwidth starvation on dual-socket AMD EPYC architectures.
