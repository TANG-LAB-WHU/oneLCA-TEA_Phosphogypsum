# Models Directory

This directory stores pre-trained machine learning models, weights, and serialized objects used throughout the PG-LCA-TEA framework.

## Directory Structure

- **`embeddings/`**:
  - Stores text embedding models (e.g., BGE, E5, or locally fine-tuned models).
  - Used by the AI knowledge engine for RAG (Retrieval-Augmented Generation) to convert literature into vector representations.

- **`gap_filler/`**:
  - Stores ML models (e.g., XGBoost, Random Forest) used for data imputation.
  - Used to predict missing LCA/TEA parameters based on the existing knowledge base.

## Usage Note

Large model files (e.g., `.bin`, `.pt`, `.safetensors`) should be placed in their respective subdirectories. If using cloud-hosted models, this directory can also store local adapter weights or quantization configs.
