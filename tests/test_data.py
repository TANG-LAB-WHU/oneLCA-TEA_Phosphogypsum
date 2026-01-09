"""Tests for Data Layer"""
import pytest
from pgloop.iodata.pdf_parser import PDFParser
from pgloop.iodata.data_standardizer import DataStandardizer

def test_pdf_parser_init():
    parser = PDFParser()
    assert parser is not None

def test_standardizer_units():
    ds = DataStandardizer()
    val = ds.convert_unit(1.0, "tonne", "kg")
    assert val == 1000.0
