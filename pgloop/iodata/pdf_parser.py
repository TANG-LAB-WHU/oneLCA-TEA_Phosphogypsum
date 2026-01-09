"""
PDF Parser Module

Extracts text and data from scientific papers using PyMuPDF or MinerU.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


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
    - MinerU output (markdown) for pre-parsed documents
    """
    
    def __init__(self, parser_type: str = "pymupdf"):
        """
        Initialize the PDF parser.
        
        Args:
            parser_type: Type of parser to use ("pymupdf" or "mineru")
        """
        self.parser_type = parser_type
        
        if parser_type == "pymupdf" and not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
    
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
            metadata=metadata
        )
    
    def _parse_mineru_output(self, filepath: Path) -> ParsedDocument:
        """Parse MinerU markdown output."""
        # Look for corresponding .md file
        md_path = filepath.with_suffix(".md")
        
        if not md_path.exists():
            # Try common MinerU naming patterns
            parent = filepath.parent
            stem = filepath.stem
            patterns = [
                parent / f"MinerU_markdown_{stem}.md",
                parent / "Papers_Parsed" / f"MinerU_markdown_{stem}.md",
            ]
            
            for pattern in patterns:
                if pattern.exists():
                    md_path = pattern
                    break
            else:
                raise FileNotFoundError(f"MinerU output not found for: {filepath}")
        
        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Extract title from first heading
        lines = text.split("\n")
        title = filepath.stem
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break
        
        return ParsedDocument(
            filepath=str(filepath),
            title=title,
            text=text,
            pages=-1,  # Not applicable for markdown
            tables=[],
            metadata={"source": "mineru"}
        )
    
    def parse_directory(
        self, 
        directory: Union[str, Path],
        pattern: str = "*.pdf"
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
            except Exception as e:
                print(f"Error parsing {pdf_path}: {e}")
        
        return documents


if __name__ == "__main__":
    # Example usage
    parser = PDFParser(parser_type="pymupdf")
    # doc = parser.parse_pdf("path/to/paper.pdf")
    # print(doc.title)
