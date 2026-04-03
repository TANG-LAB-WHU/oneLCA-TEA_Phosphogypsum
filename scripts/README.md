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

- **Project extras (recommended)**: `pip install -e ".[rag,pdf]"`
- **MinerU**: Advanced PDF parsing with OCR (`pip install -U "mineru[all]"`)
- **LightRAG**: Graph-enhanced RAG (`pip install lightrag-hku`)
- **RAGAnything**: Multimodal RAG (optional, `pip install "raganything[all]"`)

### Environment Variables

Configure in `.env`:

```env
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen3.5:35b
EMBEDDING_MODEL=bge-m3:567m
EMBEDDING_DIM=1024
```
