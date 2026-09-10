"""
Data Ingestion Layer

This module handles data collection from various sources:
- PDF parsing (local papers)
- Web scraping (open access databases)
- API connectors (regulatory databases)
- Data standardization
"""

from pgloop.iodata.api_connector import APIConnector
from pgloop.iodata.data_standardizer import DataStandardizer
from pgloop.iodata.pdf_parser import PDFParser
from pgloop.iodata.web_scraper import WebScraper

try:
    from pgloop.iodata.edge_bridge import EdgeBridge
except ImportError:
    EdgeBridge = None

try:
    from pgloop.iodata.stream_processor import StreamProcessor
except ImportError:
    StreamProcessor = None

from pgloop.iodata.datahub import DataHub

__all__ = [
    "DataHub",
    "PDFParser",
    "WebScraper",
    "DataStandardizer",
    "APIConnector",
    "EdgeBridge",
    "StreamProcessor",
]
