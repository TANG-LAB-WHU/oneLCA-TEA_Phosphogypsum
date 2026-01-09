# Data Storage Directory

This directory is used for storing input data files, intermediate results, and the persistent knowledge graph.

## Subdirectories

- **`raw/`**:
  - Place raw literature (PDFs), spreadsheets (Excel/CSV), or database extracts here.
  - This is the source for the data ingestion layer.

- **`processed/`**:
  - Contains standardized data, parsed markdown files, and the saved state of the Knowledge Graph (e.g., NetworkX pickles or Neo4j data exports).

- **`templates/`**:
  - Provides standardized templates for data entry to ensure compatibility with the `pgloop.iodata` ingestion tools.

## Note on Version Control

Typically, large raw data files are excluded from Git (see `.gitignore`). Only small sample files and templates should be committed to and tracked in the repository.
