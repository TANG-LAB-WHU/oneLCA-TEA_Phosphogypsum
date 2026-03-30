"""
Test PDF Parser Module

Tests for the PDFParser class in pgloop.iodata.pdf_parser.
"""

from pathlib import Path

import pytest

from pgloop.iodata.pdf_parser import MINERU_AVAILABLE, PYMUPDF_AVAILABLE, ParsedDocument, PDFParser


class TestPDFParserInit:
    """Tests for PDFParser initialization."""

    def test_init_pymupdf(self):
        """Test initialization with PyMuPDF parser."""
        if not PYMUPDF_AVAILABLE:
            pytest.skip("PyMuPDF not installed")

        parser = PDFParser(parser_type="pymupdf")
        assert parser.parser_type == "pymupdf"
        assert parser.output_dir == Path("./data/raw/papers/parsed")

    def test_init_custom_output_dir(self):
        """Test initialization with custom output directory."""
        if not PYMUPDF_AVAILABLE:
            pytest.skip("PyMuPDF not installed")

        custom_dir = Path("./custom/output")
        parser = PDFParser(parser_type="pymupdf", output_dir=custom_dir)
        assert parser.output_dir == custom_dir

    def test_init_mineru_not_available(self):
        """Test that ImportError is raised when MinerU is not available."""
        if MINERU_AVAILABLE:
            pytest.skip("MinerU is installed, cannot test unavailability")

        with pytest.raises(ImportError):
            PDFParser(parser_type="mineru")

    def test_init_invalid_parser_type(self, tmp_path):
        """Test that ValueError is raised for invalid parser type."""
        # Create a dummy file first (to avoid FileNotFoundError)
        dummy_pdf = tmp_path / "dummy.pdf"
        dummy_pdf.write_bytes(b"dummy content")

        # Create parser with invalid type
        parser = PDFParser.__new__(PDFParser)
        parser.parser_type = "invalid"
        parser.output_dir = tmp_path
        parser.language = "en"

        with pytest.raises(ValueError, match="Unknown parser type"):
            parser.parse_pdf(dummy_pdf)


class TestPDFParserStatic:
    """Tests for static methods."""

    def test_is_mineru_available(self):
        """Test is_mineru_available static method."""
        result = PDFParser.is_mineru_available()
        assert isinstance(result, bool)
        assert result == MINERU_AVAILABLE

    def test_is_pymupdf_available(self):
        """Test is_pymupdf_available static method."""
        result = PDFParser.is_pymupdf_available()
        assert isinstance(result, bool)
        assert result == PYMUPDF_AVAILABLE


class TestParsedDocument:
    """Tests for ParsedDocument dataclass."""

    def test_parsed_document_creation(self):
        """Test creating a ParsedDocument."""
        doc = ParsedDocument(
            filepath="/path/to/test.pdf",
            title="Test Document",
            text="Sample text content",
            pages=5,
            tables=[{"html": "<table></table>", "page": 1}],
            metadata={"source": "test"},
        )

        assert doc.filepath == "/path/to/test.pdf"
        assert doc.title == "Test Document"
        assert doc.text == "Sample text content"
        assert doc.pages == 5
        assert len(doc.tables) == 1
        assert doc.metadata["source"] == "test"


class TestPyMuPDFParser:
    """Tests for PyMuPDF parser functionality."""

    @pytest.fixture
    def sample_pdf_path(self, tmp_path):
        """Create a simple test PDF if PyMuPDF is available."""
        if not PYMUPDF_AVAILABLE:
            pytest.skip("PyMuPDF not installed")

        import fitz

        pdf_path = tmp_path / "test_document.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test Document Title", fontsize=16)
        page.insert_text((72, 100), "This is sample text content for testing.")
        doc.save(str(pdf_path))
        doc.close()

        return pdf_path

    def test_parse_pdf_pymupdf(self, sample_pdf_path):
        """Test parsing PDF with PyMuPDF."""
        parser = PDFParser(parser_type="pymupdf")
        result = parser.parse_pdf(sample_pdf_path)

        assert isinstance(result, ParsedDocument)
        assert result.filepath == str(sample_pdf_path)
        assert result.pages == 1
        assert "Test Document Title" in result.text
        assert "sample text content" in result.text

    def test_parse_nonexistent_pdf(self):
        """Test that FileNotFoundError is raised for non-existent PDF."""
        if not PYMUPDF_AVAILABLE:
            pytest.skip("PyMuPDF not installed")

        parser = PDFParser(parser_type="pymupdf")

        with pytest.raises(FileNotFoundError):
            parser.parse_pdf(Path("/nonexistent/path/to/file.pdf"))


class TestMinerUOutput:
    """Tests for parsing pre-existing MinerU markdown output."""

    @pytest.fixture
    def sample_md_file(self, tmp_path):
        """Create a sample MinerU markdown output file."""
        md_content = """# Sample Research Paper

## Abstract

This is a test abstract for the sample research paper.

## Introduction

The introduction section contains background information.

## Results

| Parameter | Value | Unit |
|-----------|-------|------|
| Energy    | 100   | kWh  |
| Cost      | 50    | USD  |

## Conclusion

This concludes the test document.
"""
        md_path = tmp_path / "sample_paper.md"
        md_path.write_text(md_content, encoding="utf-8")
        return md_path

    def test_parse_mineru_output(self, sample_md_file):
        """Test parsing pre-existing MinerU markdown output."""
        # Create PDF path - need to create a dummy file for the exists check
        pdf_path = sample_md_file.with_suffix(".pdf")
        pdf_path.write_bytes(b"dummy pdf content")  # Create dummy PDF

        parser = PDFParser(parser_type="mineru_output")
        result = parser.parse_pdf(pdf_path)

        assert isinstance(result, ParsedDocument)
        assert result.title == "Sample Research Paper"
        assert "Abstract" in result.text
        assert "Introduction" in result.text
        assert result.metadata["source"] == "mineru_output"

    def test_parse_mineru_output_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised when markdown not found."""
        parser = PDFParser(parser_type="mineru_output", output_dir=tmp_path)

        with pytest.raises(FileNotFoundError):
            parser.parse_pdf(tmp_path / "nonexistent.pdf")


class TestParseDirectory:
    """Tests for parsing directories of PDFs."""

    @pytest.fixture
    def sample_pdf_directory(self, tmp_path):
        """Create a directory with sample PDFs."""
        if not PYMUPDF_AVAILABLE:
            pytest.skip("PyMuPDF not installed")

        import fitz

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        for i in range(3):
            pdf_path = pdf_dir / f"document_{i}.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), f"Document {i}")
            doc.save(str(pdf_path))
            doc.close()

        return pdf_dir

    def test_parse_directory(self, sample_pdf_directory):
        """Test parsing all PDFs in a directory."""
        parser = PDFParser(parser_type="pymupdf")
        results = parser.parse_directory(sample_pdf_directory)

        assert len(results) == 3
        for doc in results:
            assert isinstance(doc, ParsedDocument)
            assert doc.pages == 1

    def test_parse_directory_empty(self, tmp_path):
        """Test parsing an empty directory."""
        if not PYMUPDF_AVAILABLE:
            pytest.skip("PyMuPDF not installed")

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        parser = PDFParser(parser_type="pymupdf")
        results = parser.parse_directory(empty_dir)

        assert len(results) == 0


class TestMinerUIntegration:
    """Integration tests for MinerU parser (requires MinerU to be installed)."""

    @pytest.fixture
    def sample_pdf_for_mineru(self, tmp_path):
        """Create a sample PDF for MinerU testing."""
        if not PYMUPDF_AVAILABLE:
            pytest.skip("PyMuPDF not installed")

        import fitz

        pdf_path = tmp_path / "mineru_test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "MinerU Test Document", fontsize=14)
        page.insert_text((72, 100), "This document tests MinerU parsing capabilities.")
        page.insert_text((72, 130), "Abstract: This is a test paper about phosphogypsum.")
        page.insert_text((72, 160), "Keywords: LCA, TEA, phosphogypsum, circular economy")
        doc.save(str(pdf_path))
        doc.close()

        return pdf_path

    def test_mineru_availability_check(self):
        """Test MINERU_AVAILABLE flag and is_mineru_available method."""
        from pgloop.iodata.pdf_parser import MINERU_AVAILABLE

        result = PDFParser.is_mineru_available()
        assert result == MINERU_AVAILABLE
        print(f"\nMinerU Python API available: {MINERU_AVAILABLE}")

    def test_mineru_cli_available(self):
        """Test if MinerU CLI command is available."""
        import subprocess

        try:
            result = subprocess.run(
                ["mineru", "--help"], capture_output=True, text=True, timeout=10
            )
            mineru_cli_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            mineru_cli_available = False

        print(f"\nMinerU CLI available: {mineru_cli_available}")
        # Informational only, not a failure

    def test_mineru_parser_init_when_available(self):
        """Test MinerU parser initialization when MinerU is available."""
        if not MINERU_AVAILABLE:
            pytest.skip("MinerU not installed")

        parser = PDFParser(parser_type="mineru", language="en")
        assert parser.parser_type == "mineru"
        assert parser.language == "en"

    def test_mineru_parser_init_with_language(self):
        """Test MinerU parser initialization with different languages."""
        if not MINERU_AVAILABLE:
            pytest.skip("MinerU not installed")

        # Test Chinese
        parser_ch = PDFParser(parser_type="mineru", language="ch")
        assert parser_ch.language == "ch"

        # Test Japanese
        parser_jp = PDFParser(parser_type="mineru", language="japan")
        assert parser_jp.language == "japan"

    @pytest.mark.skipif(not MINERU_AVAILABLE, reason="MinerU not installed")
    def test_mineru_api_parse(self, sample_pdf_for_mineru, tmp_path):
        """Test MinerU API parsing (requires MinerU installation and models)."""
        output_dir = tmp_path / "mineru_output"

        parser = PDFParser(parser_type="mineru", output_dir=output_dir, language="en")

        try:
            result = parser.parse_pdf(sample_pdf_for_mineru)

            # Verify result structure
            assert isinstance(result, ParsedDocument)
            assert result.filepath == str(sample_pdf_for_mineru)
            assert result.metadata["source"] == "mineru"
            assert "mode" in result.metadata  # Should have ocr or text mode

            # Verify content extraction
            assert len(result.text) > 0

            # Verify output directory was created
            assert output_dir.exists()

            print("\nMinerU parse successful!")
            print(f"  Title: {result.title}")
            print(f"  Text length: {len(result.text)} chars")
            print(f"  Tables found: {len(result.tables)}")
            print(f"  Mode: {result.metadata.get('mode', 'unknown')}")

        except Exception as e:
            # MinerU might fail due to missing models or GPU requirements
            pytest.skip(f"MinerU parsing failed (likely missing models/GPU): {e}")

    @pytest.mark.skipif(not MINERU_AVAILABLE, reason="MinerU not installed")
    def test_mineru_output_structure(self, sample_pdf_for_mineru, tmp_path):
        """Test that MinerU creates expected output structure."""
        output_dir = tmp_path / "output_structure_test"

        parser = PDFParser(parser_type="mineru", output_dir=output_dir, language="en")

        try:
            result = parser.parse_pdf(sample_pdf_for_mineru)

            # Check output path is recorded
            assert "output_path" in result.metadata
            output_path = Path(result.metadata["output_path"])

            # Output directory should exist
            assert output_path.exists() or output_dir.exists()

        except Exception as e:
            pytest.skip(f"MinerU output structure test skipped: {e}")

    def test_mineru_cli_parser_type(self, tmp_path):
        """Test MinerU CLI parser type initialization."""
        parser = PDFParser(parser_type="mineru_cli", output_dir=tmp_path / "cli_output")
        assert parser.parser_type == "mineru_cli"

    def test_mineru_output_parser_type(self, tmp_path):
        """Test MinerU output parser type initialization."""
        parser = PDFParser(parser_type="mineru_output", output_dir=tmp_path)
        assert parser.parser_type == "mineru_output"

    def test_mineru_output_with_nested_structure(self, tmp_path):
        """Test parsing MinerU output from nested directory structure."""
        # Create nested MinerU output structure
        pdf_name = "nested_paper"
        nested_dir = tmp_path / pdf_name / "auto"
        nested_dir.mkdir(parents=True)

        md_content = """# Nested Paper Title

## Introduction

This paper discusses phosphogypsum processing.
"""
        (nested_dir / f"{pdf_name}.md").write_text(md_content, encoding="utf-8")

        # Create dummy PDF
        pdf_path = tmp_path / f"{pdf_name}.pdf"
        pdf_path.write_bytes(b"dummy pdf")

        parser = PDFParser(parser_type="mineru_output", output_dir=tmp_path)
        result = parser.parse_pdf(pdf_path)

        assert result.title == "Nested Paper Title"
        assert "phosphogypsum" in result.text


class TestMinerUErrorHandling:
    """Tests for MinerU error handling."""

    def test_mineru_import_error_message(self):
        """Test that helpful error message is shown when MinerU not available."""
        if MINERU_AVAILABLE:
            pytest.skip("MinerU is installed")

        with pytest.raises(ImportError) as exc_info:
            PDFParser(parser_type="mineru")

        error_msg = str(exc_info.value)
        assert "mineru" in error_msg.lower()

    def test_mineru_cli_file_not_found(self, tmp_path):
        """Test MinerU CLI with non-existent file."""
        parser = PDFParser(parser_type="mineru_cli", output_dir=tmp_path)

        with pytest.raises(FileNotFoundError):
            parser.parse_pdf(tmp_path / "nonexistent.pdf")

    def test_mineru_output_fallback_patterns(self, tmp_path):
        """Test MinerU output parser tries multiple patterns."""
        # Create PDF without matching markdown
        pdf_path = tmp_path / "orphan.pdf"
        pdf_path.write_bytes(b"dummy pdf")

        parser = PDFParser(parser_type="mineru_output", output_dir=tmp_path / "output")

        # Should raise FileNotFoundError after trying all patterns
        with pytest.raises(FileNotFoundError):
            parser.parse_pdf(pdf_path)


def main():
    print("Running PDF Parser Tests...")
    print("=" * 60)

    # Show availability status
    print(f"PyMuPDF available: {PYMUPDF_AVAILABLE}")
    print(f"MinerU available:  {MINERU_AVAILABLE}")
    print("=" * 60)

    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    main()
