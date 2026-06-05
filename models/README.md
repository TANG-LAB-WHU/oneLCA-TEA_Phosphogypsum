# Models Directory

This directory is the designated storage space for pre-trained machine learning models, weights, and serialized configuration files used throughout the **PhosphogypsumBot** framework, especially during offline HPC deployments (such as WHU-SCC).

To prevent Git repository bloat, large binary weight files (e.g., `.bin`, `.pt`, `.gguf`, `.safetensors`) should **never** be committed to Git. Instead, download them locally and place them in their respective subdirectories as defined below.

---

## 📂 Subdirectory Architecture

### 1. `models/gguf/`
*   **Purpose**: Stores GGUF-quantized weights for the Reasoner LLM and Parser VLM used by the `llama.cpp` server backends.
*   **Expected Files**:
    *   `Qwen3.6-27B-Q8_0.gguf` (Reasoner LLM)
    *   `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` (Parser VLM) + `mmproj-BF16.gguf` (VLM Vision Projector)
*   **Active Integration**: Reference these paths inside your `.env` configuration file and Slurm runscripts (`slurm_jobs/`).

### 2. `models/embeddings/`
*   **Purpose**: Stores local text embedding models for LightRAG and vector RAG indices.
*   **Active Integration**: Loaded by the RAG orchestration engines in [pgloop/knowledge/](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/knowledge/) (e.g., `LightRAGEngine` or `RAGAnythingEngine`).

### 3. `models/mace/`
*   **Purpose**: Stores serialized pre-trained universal interatomic force field weights (like MACE-MP-Medium) for chemical property prediction.
*   **Active Integration**: Dynamically loaded by `MACEPredictor` in [pgloop/chemicals/property_predictor.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/chemicals/property_predictor.py).

### 4. `models/mineru/`
*   **Purpose**: Stores the layout parsing, formula recognition, and OCR neural network weights for MinerU.
*   **Active Integration**: Used by `PDFParser` in [pgloop/iodata/pdf_parser.py](file:///Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum/pgloop/iodata/pdf_parser.py). For offline supercomputer nodes, copy these model assets directly here and override the MinerU config schema paths.

---

## ⚠️ Git Enforcement Rule
All subdirectories are ignored by Git (configured via the root `.gitignore`) except for their respective `.gitkeep` files. Ensure you download and copy weights manually during cluster environment initialization.
