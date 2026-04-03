# Scripts Directory

This directory contains utility scripts for building and managing the PG-LCA-TEA knowledge base.

---

## `build_knowledge_graph.py`

Main pipeline script for processing phosphogypsum literature and building the knowledge graph.

### Usage

```bash
python scripts/build_knowledge_graph.py [OPTIONS]
```

### Options

| Option | Choices | Default | Description |
|--------|---------|---------|-------------|
| `--step` | `all`, `parse`, `index`, `extract`, `build` | `all` | Pipeline step to run |
| `--parser` | `pymupdf`, `mineru` | `mineru` | PDF parser to use |
| `--limit` | integer | None | Limit documents to process (for testing) |
| `--engine` | `lightrag`, `raganything` | `lightrag` | RAG engine to use |

### Pipeline Steps

| Step | Description | Output |
|------|-------------|--------|
| `parse` | Extract text from PDFs using MinerU or PyMuPDF | `data/raw/papers/parsed/*.md` |
| `index` | Build RAG index with entity extraction | `data/processed/lightrag_db/` or `data/processed/raganything_db/` |
| `extract` | Extract structured data using LLM | `data/processed/extracted_data/*.json` |
| `build` | Construct domain knowledge graph | `data/processed/knowledge_graph/` |

### Examples

```bash
# Run full pipeline with defaults (MinerU + LightRAG)
python scripts/build_knowledge_graph.py

# Parse 3 PDFs for testing
python scripts/build_knowledge_graph.py --step parse --limit 3

# Build index using RAGAnything (multimodal enhanced)
python scripts/build_knowledge_graph.py --step index --engine raganything

# Run RAGAnything on 2 PDFs for testing
python scripts/build_knowledge_graph.py --engine raganything --limit 2
```

---

## Architecture

### Parser Comparison

| Parser | Speed | Images | Tables | Formulas | OCR |
|--------|-------|--------|--------|----------|-----|
| `pymupdf` | Fast | ❌ | ❌ | ❌ | ❌ |
| `mineru` | Slow | ✅ | ✅ HTML | ✅ LaTeX | ✅ |

### Engine Comparison

| Engine | Source | Features | API Cost |
|--------|--------|----------|----------|
| `lightrag` | Pre-parsed `.md` | Entity extraction, KG indexing | LLM + Vision (5 images/doc) |
| `raganything` | Raw PDF | Unified multimodal, better table/formula | LLM + Vision (all images) |

### Recommended Combinations

| Scenario | RAG Engine | Knowledge Graph Backend |
|----------|------------|-------------------------|
| Development & Testing | LightRAG | NetworkX |
| Multimodal Processing | RAGAnything | NetworkX |
| Production Deployment | RAGAnything | Neo4j |

### Data Flow

```
data/raw/papers/unparsed/*.pdf
        ↓ (parse step - MinerU)
data/raw/papers/parsed/
    └── paper_name/
        ├── paper_name.md      ← Markdown with text, tables, formulas
        └── auto/images/       ← Extracted figures
        ↓ (index step - LightRAG/RAGAnything)
data/processed/lightrag_db/    ← Entity-relationship graph + vector index
        ↓ (extract step - LLM)
data/processed/extracted_data/ ← Structured JSON (compositions, technologies, LCI, costs)
        ↓ (build step)
data/processed/knowledge_graph/ ← Domain knowledge graph (NetworkX/Neo4j)
```

---

## Dependencies

- **Project extras (recommended)**: `pip install -e ".[rag]"` (MinerU is a core dependency)
- **MinerU**: Advanced PDF parsing with OCR (`pip install -U "mineru[all]"`)
- **LightRAG**: Graph-enhanced RAG (`pip install lightrag-hku`)
- **RAGAnything**: Multimodal RAG (optional, `pip install "raganything[all]"`)

### Environment Variables

Configure in `.env`:

```env
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen3.5:35b
EMBEDDING_MODEL=qwen3-embedding:4b
EMBEDDING_DIM=2560
LLM_CONTEXT_LENGTH=32768
LLM_JSON_MODE=1
MINERU_MODEL_SOURCE=huggingface
HF_ENDPOINT=https://huggingface.co
# Optional MinerU overrides (used by RAGAnythingEngine when set)
# RAGANYTHING_MINERU_BACKEND=pipeline
# RAGANYTHING_MINERU_DEVICE=cpu
```

On your deployed Windows desktop, verify embedding dimension once after pulling the model:

```powershell
python -c "from openai import OpenAI; import os; from dotenv import load_dotenv; load_dotenv(); c=OpenAI(base_url=os.getenv('LLM_BASE_URL'), api_key=os.getenv('LLM_API_KEY','ollama')); v=c.embeddings.create(model=os.getenv('EMBEDDING_MODEL'), input=['hello']).data[0].embedding; print('dim=',len(v))"
```

If this prints a value other than `2560`, set `.env` `EMBEDDING_DIM` to that value.

### Ollama Stability Checklist (Windows)

For local `qwen3.5:35b` + `qwen3-embedding:4b`, set these as **OS-level environment variables**, then restart Ollama:

```powershell
setx OLLAMA_FLASH_ATTENTION 0
setx OLLAMA_CONTEXT_LENGTH 32768
setx OLLAMA_NUM_PARALLEL 1
setx OLLAMA_MAX_LOADED_MODELS 1
```

Then fully quit Ollama from tray icon and launch it again.

### Optional: create a 32k model alias via Modelfile

If your Ollama version ignores request-level context hints, create a model alias with explicit `num_ctx`:

```text
# Modelfile.qwen35-32k
FROM qwen3.5:35b
PARAMETER num_ctx 32768
```

```powershell
ollama create qwen3.5:35b-32k -f Modelfile.qwen35-32k
```

Then set `LLM_MODEL=qwen3.5:35b-32k` in `.env`.

### MinerU preflight (recommended before RAGAnything)

Run a single-file parse before full pipeline:

```bash
mineru -p "<path-to-one-small-pdf>" -o "<output-dir>" -m auto
```

If model fetching fails:

- this repo defaults to Hugging Face official source:
  - `MINERU_MODEL_SOURCE=huggingface`
  - `HF_ENDPOINT=https://huggingface.co`
- set `MINERU_MODEL_SOURCE=modelscope` only when HF is blocked/slow
- set `HF_TOKEN` if your environment requires authentication
- on Windows, enabling Developer Mode avoids symlink-related cache limitations
