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

# Troubleshoot: Run RAGAnything with mirror and offline fix (Windows)
$env:HF_ENDPOINT = "https://hf-mirror.com"; $env:TRANSFORMERS_OFFLINE = 0; $env:HF_HUB_OFFLINE = 0; python scripts/build_knowledge_graph.py --engine raganything --limit 2
```

> [!TIP]
> **Windows Developer Mode**: Keep "Developer Mode" enabled in Windows settings to avoid 0-byte symlink issues with Hugging Face models. If you prefer to keep it off, consider copying the model files directly to the project directory.

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

- **MinerU**: Advanced PDF parsing with OCR (`pip install -U "mineru[all]"`)
- **LightRAG**: Graph-enhanced RAG (`pip install lightrag-hku`)
- **RAGAnything**: Multimodal RAG (optional, `pip install "raganything[all]"`)

### Environment Variables

Configure in `.env`:

```env
LLM_BASE_URL=http://127.0.0.1:8045/v1
LLM_API_KEY=your_api_key
LLM_MODEL=gemini-3-flash
GEMINI_API_KEY=your_gemini_key
EMBEDDING_MODEL=bge-m3
```
