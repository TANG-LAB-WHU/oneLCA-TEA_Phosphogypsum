"""
Data Ingestion Layer

This module handles data collection from various sources:
- PDF parsing (local papers)
- Web scraping (open access databases)
- API connectors (regulatory databases)
- Data standardization
"""

from pgloop.iodata.pdf_parser import PDFParser
from pgloop.iodata.data_standardizer import DataStandardizer
from pgloop.iodata.api_connector import APIConnector

__all__ = ["PDFParser", "DataStandardizer", "APIConnector"]
