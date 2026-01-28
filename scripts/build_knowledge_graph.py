"""
This script processes PDF papers and builds a knowledge graph for PG-LCA-TEA.

Workflow:
1. Parse PDFs from data/raw/papers/unparsed/ → data/raw/papers/parsed/
2. Build LightRAG index (Graph + Vector) in data/processed/lightrag_db/
3. Extract structured data using Gemini LLM
4. Construct knowledge graph in data/processed/knowledge_graph/

Usage:
    python scripts/build_knowledge_graph.py --step all
    python scripts/build_knowledge_graph.py --step parse
    python scripts/build_knowledge_graph.py --step index
    python scripts/build_knowledge_graph.py --step extract
    python scripts/build_knowledge_graph.py --step build
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

from pgloop.iodata import PDFParser
from pgloop.knowledge import LightRAGEngine, LLMExtractor, PhosphogypsumKG

# Optional RAGAnything support
try:
    from pgloop.knowledge import RAGAnythingEngine, RAGANYTHING_AVAILABLE
except ImportError:
    RAGAnythingEngine = None
    RAGANYTHING_AVAILABLE = False

RAGANYTHING_DIR = PROCESSED_DIR / "raganything_db" if 'PROCESSED_DIR' in dir() else None

# === Directory Configuration ===
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

UNPARSED_DIR = RAW_DIR / "papers" / "unparsed"
PARSED_DIR = RAW_DIR / "papers" / "parsed"
LIGHTRAG_DIR = PROCESSED_DIR / "lightrag_db"
RAGANYTHING_DIR = PROCESSED_DIR / "raganything_db"
KG_DIR = PROCESSED_DIR / "knowledge_graph"
EXTRACTED_DIR = PROCESSED_DIR / "extracted_data"

# Ensure directories exist
for d in [PARSED_DIR, LIGHTRAG_DIR, RAGANYTHING_DIR, KG_DIR, EXTRACTED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def step1_parse_pdfs(parser_type: str = "pymupdf", limit: int = None):
    """
    Step 1: Parse PDFs to Markdown
    
    Args:
        parser_type: 'pymupdf' (fast) or 'mineru' (accurate tables/formulas)
        limit: Max number of PDFs to process (None for all)
    """
    print("\n" + "=" * 60)
    print("   STEP 1: PDF PARSING")
    print("=" * 60)
    
    # Initialize parser
    parser = PDFParser(
        parser_type=parser_type,
        output_dir=PARSED_DIR,
        language="en"
    )
    
    print(f"Parser type: {parser_type}")
    print(f"Input:  {UNPARSED_DIR}")
    print(f"Output: {PARSED_DIR}")
    
    # Get PDF files
    pdf_files = list(UNPARSED_DIR.glob("*.pdf"))
    if limit:
        pdf_files = pdf_files[:limit]
    
    print(f"\nFound {len(pdf_files)} PDF files to process")
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            # Check if already parsed (support both flat .md and subdirectory structure)
            output_md_flat = PARSED_DIR / f"{pdf_path.stem}.md"
            output_md_subdir = PARSED_DIR / pdf_path.stem / f"{pdf_path.stem}.md"
            
            if output_md_flat.exists() or output_md_subdir.exists():
                skipped_count += 1
                print(f"[{i}/{len(pdf_files)}] ⏭ Skipped (already parsed): {pdf_path.name[:50]}")
                continue
            
            print(f"\n[{i}/{len(pdf_files)}] Parsing: {pdf_path.name[:50]}...")
            doc = parser.parse_pdf(pdf_path)
            
            # For MinerU parser, the .md is already saved inside subdirectory by pdf_parser
            # For PyMuPDF, save as flat .md file at top level
            if parser_type == "pymupdf":
                output_path = output_md_flat
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(f"# {doc.title}\n\n")
                    f.write(f"**Source**: {pdf_path.name}\n")
                    f.write(f"**Pages**: {doc.pages}\n\n")
                    f.write("---\n\n")
                    f.write(doc.text)
                print(f"    ✓ Saved: {output_path.name} ({len(doc.text)} chars)")
            else:
                # MinerU already saved to subdirectory
                print(f"    ✓ Saved: {pdf_path.stem}/{pdf_path.stem}.md ({len(doc.text)} chars)")
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            print(f"    ✗ Error: {e}")
    
    print(f"\n{'=' * 40}")
    print(f"Parsing Complete: {success_count} success, {skipped_count} skipped, {error_count} errors")
    return success_count


def step2_build_rag_index(limit: int = None, engine: str = "lightrag"):
    """
    Step 2: Build RAG Index (Graph + Vector)
    
    Indexes parsed documents using LightRAG or RAGAnything with entity-relationship extraction.
    Note: This step consumes Gemini API calls for entity extraction.
    
    Args:
        limit: Max documents to process
        engine: "lightrag" (default) or "raganything" (multimodal enhanced)
    """
    print("\n" + "=" * 60)
    engine_upper = engine.upper()
    print(f"   STEP 2: {engine_upper} INDEX CONSTRUCTION")
    print("=" * 60)
    
    if engine == "raganything":
        if not RAGANYTHING_AVAILABLE:
            print("ERROR: RAGAnything not installed. Run: pip install 'raganything[all]'")
            print("Falling back to LightRAG...")
            engine = "lightrag"
    
    if engine == "raganything":
        # Use RAGAnything for multimodal processing
        rag = RAGAnythingEngine(
            working_dir=RAGANYTHING_DIR
        )
        working_dir = RAGANYTHING_DIR
        print(f"RAGAnything DB: {RAGANYTHING_DIR}")
        print(f"Parser:      {rag.parser}")
    else:
        # Use LightRAG (default)
        rag = LightRAGEngine(
            working_dir=LIGHTRAG_DIR
        )
        working_dir = LIGHTRAG_DIR
        print(f"LightRAG DB: {LIGHTRAG_DIR}")
    
    print(f"Source:      {PARSED_DIR}")
    print(f"LLM Model:   {rag.llm_model}")
    print("\n⚠️  Note: Entity extraction will consume Gemini API quota")
    
    # Index documents
    if engine == "raganything":
        # RAGAnything can process raw PDFs directly
        results = rag.process_documents_from_directory(
            directory=UNPARSED_DIR,
            pattern="*.pdf",
            limit=limit
        )
    else:
        # LightRAG uses pre-parsed markdown
        results = rag.add_documents_from_directory(
            directory=PARSED_DIR,
            pattern="*.md",
            limit=limit,
            transcribe_images=True
        )
    
    stats = rag.get_statistics()
    print(f"\n{'=' * 40}")
    print(f"Indexed {sum(results.values())} / {len(results)} documents")
    print(f"Working dir: {stats.get('working_dir', str(working_dir))}")
    
    return results


def step3_extract_structured_data(limit: int = None):
    """
    Step 3: Extract Structured Data using LLM
    
    Uses Gemini to extract composition, technology, LCI, and cost data.
    """
    print("\n" + "=" * 60)
    print("   STEP 3: LLM DATA EXTRACTION")
    print("=" * 60)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env file")
        return {}
    
    # Initialize LLM extractor
    extractor = LLMExtractor(
        provider="gemini",
        model="gemini-2.0-flash",
        api_key=api_key
    )
    
    print(f"LLM: Gemini (gemini-2.0-flash)")
    print(f"Source: {PARSED_DIR}")
    print(f"Output: {EXTRACTED_DIR}")
    
    # Get parsed files
    md_files = list(PARSED_DIR.glob("*.md"))
    if limit:
        md_files = md_files[:limit]
    
    print(f"\nProcessing {len(md_files)} documents...")
    
    extraction_types = ["composition", "technology", "lci", "cost"]
    all_results = {}
    
    for i, md_path in enumerate(md_files, 1):
        print(f"\n[{i}/{len(md_files)}] Extracting from: {md_path.name[:40]}...")
        
        try:
            results = extractor.extract_from_document(md_path, extraction_types)
            
            # Save extraction results
            output_path = EXTRACTED_DIR / f"{md_path.stem}_extracted.json"
            
            extracted_data = {}
            for etype, result in results.items():
                if result.success:
                    extracted_data[etype] = result.data
                    print(f"    ✓ {etype}: {len(result.data)} items")
                else:
                    print(f"    ✗ {etype}: {result.errors}")
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(extracted_data, f, indent=2, ensure_ascii=False)
            
            all_results[md_path.name] = extracted_data
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    print(f"\n{'=' * 40}")
    print(f"Extracted data from {len(all_results)} documents")
    return all_results


def step4_build_knowledge_graph():
    """
    Step 4: Build Knowledge Graph
    
    Populates the knowledge graph from extracted data.
    """
    print("\n" + "=" * 60)
    print("   STEP 4: KNOWLEDGE GRAPH CONSTRUCTION")
    print("=" * 60)
    
    # Initialize KG
    kg = PhosphogypsumKG(storage_path=KG_DIR)
    
    print(f"KG Storage: {KG_DIR}")
    print(f"Extracted Data: {EXTRACTED_DIR}")
    
    # Load extracted data
    json_files = list(EXTRACTED_DIR.glob("*_extracted.json"))
    print(f"\nLoading {len(json_files)} extraction files...")
    
    stats = {"countries": 0, "compositions": 0, "technologies": 0, "sources": 0}
    
    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Add source reference
            source_id = kg.add_source(
                title=json_path.stem.replace("_extracted", ""),
                year=2024
            )
            stats["sources"] += 1
            
            # Process composition data
            if "composition" in data and data["composition"]:
                comp_data = data["composition"]
                if isinstance(comp_data, dict):
                    comp_id = kg.add_composition(
                        name=f"comp_{json_path.stem[:20]}",
                        country=comp_data.get("country", "Unknown"),
                        CaSO4=comp_data.get("CaSO4"),
                        P2O5=comp_data.get("P2O5"),
                        F=comp_data.get("F"),
                        Ra226=comp_data.get("Ra226")
                    )
                    kg.add_source_reference(comp_id, source_id)
                    stats["compositions"] += 1
            
            # Process technology data
            if "technology" in data and data["technology"]:
                tech_data = data["technology"]
                if isinstance(tech_data, dict):
                    tech_id = kg.add_technology(
                        name=tech_data.get("name", "Unknown Technology"),
                        code=tech_data.get("code", "PG-UNK"),
                        trl=tech_data.get("trl"),
                        capacity_t_year=tech_data.get("capacity")
                    )
                    kg.add_source_reference(tech_id, source_id)
                    stats["technologies"] += 1
                    
        except Exception as e:
            print(f"  ✗ Error processing {json_path.name}: {e}")
    
    # Save KG
    kg.save_graph()
    
    kg_stats = kg.get_statistics()
    print(f"\n{'=' * 40}")
    print("Knowledge Graph Statistics:")
    print(f"  Nodes: {kg_stats.get('node_count', 0)}")
    print(f"  Edges: {kg_stats.get('edge_count', 0)}")
    print(f"  Node types: {kg_stats.get('node_types', {})}")
    
    return kg_stats


def run_pipeline(steps: list, parser_type: str = "pymupdf", limit: int = None, engine: str = "lightrag"):
    """Run the complete pipeline or specific steps."""
    
    print("\n" + "=" * 60)
    print("   PG-LCA-TEA KNOWLEDGE GRAPH PIPELINE")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Steps to run: {steps}")
    print(f"RAG Engine: {engine}")
    
    results = {}
    
    if "parse" in steps or "all" in steps:
        # Skip STEP 1 if using RAGAnything as it handles parsing internally
        if engine == "raganything":
            print("\n[SKIP] Explicit Step 1 (Parse) is skipped because RAGAnything handles parsing internally.")
        else:
            results["parse"] = step1_parse_pdfs(parser_type, limit)
    
    if "index" in steps or "all" in steps:
        results["index"] = step2_build_rag_index(limit, engine)
    
    if "extract" in steps or "all" in steps:
        results["extract"] = step3_extract_structured_data(limit)
    
    if "build" in steps or "all" in steps:
        results["build"] = step4_build_knowledge_graph()
    
    print("\n" + "=" * 60)
    print("   PIPELINE COMPLETE")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build Knowledge Graph from Phosphogypsum Literature"
    )
    parser.add_argument(
        "--step",
        choices=["all", "parse", "index", "extract", "build"],
        default="all",
        help="Pipeline step to run"
    )
    parser.add_argument(
        "--parser",
        choices=["pymupdf", "mineru"],
        default="mineru",
        help="PDF parser to use (default: mineru)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process (for testing)"
    )
    parser.add_argument(
        "--engine",
        choices=["lightrag", "raganything"],
        default="lightrag",
        help="RAG engine to use (default: lightrag)"
    )
    
    args = parser.parse_args()
    
    steps = [args.step] if args.step != "all" else ["all"]
    run_pipeline(steps, args.parser, args.limit, args.engine)


if __name__ == "__main__":
    main()
