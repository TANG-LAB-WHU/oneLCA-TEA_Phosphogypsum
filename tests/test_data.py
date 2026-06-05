import pytest

from pgloop.iodata.data_standardizer import DataStandardizer
from pgloop.iodata.pdf_parser import PYMUPDF_AVAILABLE, PDFParser


@pytest.mark.skipif(not PYMUPDF_AVAILABLE, reason="PyMuPDF not installed")
def test_pdf_parser_init():
    parser = PDFParser()
    assert parser is not None


def test_standardizer_units():
    ds = DataStandardizer()
    val = ds.convert_unit(1.0, "tonne", "kg")
    assert val == 1000.0
