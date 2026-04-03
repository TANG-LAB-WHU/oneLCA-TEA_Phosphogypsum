"""
PDF Parser Module

Extracts text and data from scientific papers using PyMuPDF or MinerU.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# MinerU: probe pipeline-only imports. Do NOT import mineru.cli.common here — it pulls
# VLM/office backends and optional deps; a partial env then looked like "MinerU not installed".
_MINERU_IMPORT_ERROR: Optional[BaseException] = None
pipeline_doc_analyze_streaming: Optional[Callable[..., None]] = None
pipeline_doc_analyze: Optional[Callable[..., Any]] = None
pipeline_result_to_middle_json: Optional[Callable[..., Any]] = None
pipeline_union_make: Optional[Callable[..., Any]] = None
FileBasedDataWriter: Any = None
MakeMode: Any = None

try:
    from mineru.backend.pipeline.model_json_to_middle_json import (
        result_to_middle_json as pipeline_result_to_middle_json,
    )
    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
        union_make as pipeline_union_make,
    )
    from mineru.data.data_reader_writer import FileBasedDataWriter
    from mineru.utils.enum_class import MakeMode
except Exception as _e:
    _MINERU_IMPORT_ERROR = _e
    MINERU_AVAILABLE = False
else:
    try:
        from mineru.backend.pipeline import pipeline_analyze as _pipeline_analyze
    except Exception as _e:
        _MINERU_IMPORT_ERROR = _e
        pipeline_doc_analyze_streaming = None
        pipeline_doc_analyze = None
    else:
        pipeline_doc_analyze_streaming = getattr(_pipeline_analyze, "doc_analyze_streaming", None)
        pipeline_doc_analyze = getattr(_pipeline_analyze, "doc_analyze", None)

    if pipeline_doc_analyze_streaming is not None:
        MINERU_AVAILABLE = True
    elif pipeline_doc_analyze is not None and pipeline_result_to_middle_json is not None:
        MINERU_AVAILABLE = True
    else:
        MINERU_AVAILABLE = False
        if _MINERU_IMPORT_ERROR is None:
            _MINERU_IMPORT_ERROR = ImportError(
                "MinerU pipeline API not found: need doc_analyze_streaming (MinerU 3.x) or "
                "doc_analyze + result_to_middle_json. Upgrade: pip install -U 'mineru[all]'"
            )


def _mineru_prepare_env(output_dir: str, pdf_file_name: str, parse_method: str) -> tuple[str, str]:
    """Same layout as mineru.cli.common.prepare_env without importing cli.common."""
    local_md_dir = str(os.path.join(output_dir, pdf_file_name, parse_method))
    local_image_dir = os.path.join(str(local_md_dir), "images")
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    return local_image_dir, local_md_dir


@dataclass
class ParsedDocument:
    """Represents a parsed PDF document."""

    filepath: str
    title: str
    text: str
    pages: int
    tables: List[Dict]
    metadata: Dict


class PDFParser:
    """
    PDF Parser for extracting text and tables from scientific papers.

    Supports:
    - PyMuPDF (fitz) for basic parsing
    - MinerU for advanced parsing with OCR and table extraction
    - MinerU CLI for command-line based parsing
    - Pre-parsed MinerU markdown output
    """

    def __init__(
        self,
        parser_type: str = "pymupdf",
        output_dir: Optional[Union[str, Path]] = None,
        language: str = "en",
    ):
        """
        Initialize the PDF parser.

        Args:
            parser_type: Type of parser to use:
                - "pymupdf": Basic text extraction (default)
                - "mineru": Advanced extraction with MinerU Python API
                - "mineru_cli": Parse using MinerU command line
                - "mineru_output": Read pre-parsed MinerU markdown files
            output_dir: Directory to save MinerU output (default: ./data/raw/papers/parsed)
            language: Document language for OCR (default: "en")
                Options: 'ch', 'en', 'korean', 'japan', 'arabic', 'latin', etc.
        """
        self.parser_type = parser_type
        self.output_dir = Path(output_dir) if output_dir else Path("./data/raw/papers/parsed")
        self.language = language

        if parser_type == "pymupdf" and not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")

        if parser_type == "mineru" and not MINERU_AVAILABLE:
            msg = (
                "MinerU Python pipeline could not be loaded (import/API mismatch or missing "
                "optional deps). Install or upgrade: pip install -U 'mineru[all]'\n"
                "Ensure model weights are available. See: https://github.com/opendatalab/MinerU"
            )
            if _MINERU_IMPORT_ERROR is not None:
                raise ImportError(msg) from _MINERU_IMPORT_ERROR
            raise ImportError(msg)

    def parse_pdf(self, filepath: Union[str, Path]) -> ParsedDocument:
        """
        Parse a PDF file and extract text content.

        Args:
            filepath: Path to the PDF file

        Returns:
            ParsedDocument with extracted content
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"PDF not found: {filepath}")

        if self.parser_type == "pymupdf":
            return self._parse_with_pymupdf(filepath)
        elif self.parser_type == "mineru":
            return self._parse_with_mineru(filepath)
        elif self.parser_type == "mineru_cli":
            return self._parse_with_mineru_cli(filepath)
        elif self.parser_type == "mineru_output":
            return self._parse_mineru_output(filepath)
        else:
            raise ValueError(f"Unknown parser type: {self.parser_type}")

    def _parse_with_pymupdf(self, filepath: Path) -> ParsedDocument:
        """Parse PDF using PyMuPDF."""
        doc = fitz.open(filepath)

        text_content = []
        for page in doc:
            text_content.append(page.get_text())

        full_text = "\n".join(text_content)

        # Extract metadata
        metadata = doc.metadata or {}
        title = metadata.get("title", filepath.stem)

        doc.close()

        return ParsedDocument(
            filepath=str(filepath),
            title=title,
            text=full_text,
            pages=len(text_content),
            tables=[],  # Table extraction requires additional processing
            metadata=metadata,
        )

    def _parse_with_mineru(self, filepath: Path) -> ParsedDocument:
        """
        Parse PDF using MinerU Python API for advanced extraction.

        Features:
        - Structure-aware text extraction
        - Table extraction (HTML format)
        - Formula extraction (LaTeX format)
        - OCR for scanned documents

        Requires: pip install -U "mineru[all]"
        """
        if not MINERU_AVAILABLE:
            raise ImportError(
                "MinerU not available. Install with: pip install -U 'mineru[all]'"
            ) from _MINERU_IMPORT_ERROR

        pdf_name = filepath.stem
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pdf_bytes = filepath.read_bytes()

        # Keep output layout aligned with the rest of the pipeline:
        # <parsed>/<pdf_name>/auto/<pdf_name>.md
        local_image_dir, local_md_dir = _mineru_prepare_env(str(self.output_dir), pdf_name, "auto")
        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)

        if pipeline_doc_analyze_streaming is not None:
            state: Dict[str, object] = {}

            def on_doc_ready(_doc_index, model_list, middle_json, ocr_enable):
                state["model_list"] = model_list
                state["middle_json"] = middle_json
                state["ocr_enable"] = ocr_enable

            pipeline_doc_analyze_streaming(
                [pdf_bytes],
                [image_writer],
                [self.language],
                on_doc_ready,
                parse_method="auto",
                formula_enable=True,
                table_enable=True,
            )
            middle_json = state.get("middle_json")
            if middle_json is None:
                raise RuntimeError(
                    "MinerU doc_analyze_streaming finished without emitting a document "
                    "(empty PDF or pipeline error)."
                )
            _ocr_enable = bool(state.get("ocr_enable", False))
            _lang = self.language
        else:
            assert pipeline_doc_analyze is not None and pipeline_result_to_middle_json is not None
            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
                pipeline_doc_analyze(
                    [pdf_bytes],
                    [self.language],
                    parse_method="auto",
                    formula_enable=True,
                    table_enable=True,
                )
            )
            model_list = infer_results[0]
            images_list = all_image_lists[0]
            pdf_doc = all_pdf_docs[0]
            _lang = lang_list[0]
            _ocr_enable = ocr_enabled_list[0]
            middle_json = pipeline_result_to_middle_json(
                model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, True
            )

        pdf_info = middle_json["pdf_info"]
        image_dir_name = os.path.basename(local_image_dir)

        md_content = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir_name)
        md_writer.write_string(f"{pdf_name}.md", md_content)

        tables = []
        try:
            content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir_name)
            if isinstance(content_list, list):
                for item in content_list:
                    if isinstance(item, dict) and item.get("type") == "table":
                        tables.append(
                            {"html": item.get("content", ""), "page": item.get("page_idx", -1)}
                        )
        except Exception:
            pass

        title = pdf_name
        for line in md_content.split("\n"):
            if line.startswith("# "):
                title = line[2:].strip()
                break

        return ParsedDocument(
            filepath=str(filepath),
            title=title,
            text=md_content,
            pages=len(pdf_info) if isinstance(pdf_info, list) else -1,
            tables=tables,
            metadata={
                "source": "mineru",
                "output_path": str(local_md_dir),
                "mode": "ocr" if _ocr_enable else "text",
                "language": _lang,
            },
        )

    def _parse_with_mineru_cli(self, filepath: Path, backend: str = "pipeline") -> ParsedDocument:
        """
        Parse PDF using MinerU command line interface.

        This method uses the 'mineru' CLI command which may be more stable
        in some environments.

        Args:
            filepath: Path to the PDF file
            backend: MinerU backend to use:
                - "pipeline": General purpose, CPU-friendly
                - "hybrid-auto-engine": High accuracy (requires GPU)
                - "vlm-auto-engine": Vision-Language Model based

        Returns:
            ParsedDocument with extracted content
        """
        pdf_name = filepath.stem
        output_path = self.output_dir / pdf_name
        output_path.mkdir(parents=True, exist_ok=True)

        cmd = ["mineru", "-p", str(filepath), "-o", str(output_path), "-b", backend]

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"MinerU CLI failed: {e.stderr}")
        except FileNotFoundError:
            raise ImportError(
                "mineru command not found. Install with: pip install -U 'mineru[all]'"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"MinerU parsing timed out for: {filepath}")

        md_files = list(output_path.rglob("*.md"))
        if not md_files:
            raise FileNotFoundError(f"No markdown output found in {output_path}")

        md_path = md_files[0]
        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()

        title = pdf_name
        for line in text.split("\n"):
            if line.startswith("# "):
                title = line[2:].strip()
                break

        return ParsedDocument(
            filepath=str(filepath),
            title=title,
            text=text,
            pages=-1,
            tables=[],
            metadata={"source": "mineru_cli", "output_path": str(output_path), "backend": backend},
        )

    def _parse_mineru_output(self, filepath: Path) -> ParsedDocument:
        """Parse pre-existing MinerU markdown output."""
        md_path = filepath.with_suffix(".md")

        if not md_path.exists():
            stem = filepath.stem
            patterns = [
                filepath.parent / f"{stem}.md",
                self.output_dir / stem / f"{stem}.md",
                self.output_dir / stem / "auto" / f"{stem}.md",
                self.output_dir / stem / stem / "auto" / f"{stem}.md",
            ]

            for pattern in patterns:
                if pattern.exists():
                    md_path = pattern
                    break
            else:
                raise FileNotFoundError(f"MinerU output not found for: {filepath}")

        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()

        title = filepath.stem
        for line in text.split("\n"):
            if line.startswith("# "):
                title = line[2:].strip()
                break

        return ParsedDocument(
            filepath=str(filepath),
            title=title,
            text=text,
            pages=-1,
            tables=[],
            metadata={"source": "mineru_output"},
        )

    def parse_directory(
        self, directory: Union[str, Path], pattern: str = "*.pdf"
    ) -> List[ParsedDocument]:
        """
        Parse all PDF files in a directory.

        Args:
            directory: Directory containing PDF files
            pattern: Glob pattern for file matching

        Returns:
            List of ParsedDocument objects
        """
        directory = Path(directory)
        documents = []

        for pdf_path in directory.glob(pattern):
            try:
                doc = self.parse_pdf(pdf_path)
                documents.append(doc)
                print(f"✓ Parsed: {pdf_path.name}")
            except Exception as e:
                print(f"✗ Error parsing {pdf_path.name}: {e}")

        return documents

    @staticmethod
    def is_mineru_available() -> bool:
        """Check if MinerU is available for use."""
        return MINERU_AVAILABLE

    @staticmethod
    def is_pymupdf_available() -> bool:
        """Check if PyMuPDF is available for use."""
        return PYMUPDF_AVAILABLE


def main():
    print("PDF Parser Module")
    print("-" * 40)
    print(f"PyMuPDF available: {PYMUPDF_AVAILABLE}")
    print(f"MinerU available:  {MINERU_AVAILABLE}")
    print()
    print("Usage examples:")
    print()
    print("# Basic parsing with PyMuPDF")
    print("parser = PDFParser(parser_type='pymupdf')")
    print("doc = parser.parse_pdf('paper.pdf')")
    print()
    print("# Advanced parsing with MinerU")
    print("parser = PDFParser(parser_type='mineru', language='en')")
    print("doc = parser.parse_pdf('paper.pdf')")
    print()
    print("# CLI-based parsing with MinerU")
    print("parser = PDFParser(parser_type='mineru_cli')")
    print("doc = parser.parse_pdf('paper.pdf')")


if __name__ == "__main__":
    main()
