# Local Papers Directory

This directory contains literature papers for the PG-LCA-TEA knowledge extraction pipeline.

## Directory Structure

```
papers/
├── unparsed/     # Original PDF papers (raw input)
├── parsed/       # MinerU-converted markdown documents
└── README.md     # This file
```

## Usage

### `unparsed/`
Place original PDF papers here. These will be processed by the data ingestion pipeline:
- PDF Parser (MinerU/PyMuPDF) extracts text and tables
- Output is saved to the `parsed/` directory

### `parsed/`
Contains markdown documents converted from PDFs using tools like [MinerU](https://github.com/opendatalab/MinerU).

The LLM-RAG pipeline reads from this directory to:
- Chunk and embed text for vector search
- Extract structured LCA/TEA data using LLM
- Build the Knowledge Graph

## Supported Formats

| Directory  | Formats                                |
|------------|----------------------------------------|
| `unparsed` | `.pdf`                                 |
| `parsed`   | `.md`, `.json` (MinerU output format)  |
