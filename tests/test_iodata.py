"""
Test Data Ingestion Layer

Tests for the iodata module: PDFParser, WebScraper, APIConnector, DataStandardizer.
"""

from unittest.mock import MagicMock, patch

import pytest

from pgloop.iodata import APIConnector, DataStandardizer, PDFParser, WebScraper


class TestWebScraper:
    """Tests for WebScraper functionality."""

    def test_init(self):
        """Test WebScraper initialization."""
        scraper = WebScraper()
        assert "User-Agent" in scraper.headers

    def test_custom_user_agent(self):
        """Test WebScraper with custom user agent."""
        custom_agent = "CustomBot/1.0"
        scraper = WebScraper(user_agent=custom_agent)
        assert scraper.headers["User-Agent"] == custom_agent

    def test_extract_text(self):
        """Test HTML text extraction."""
        scraper = WebScraper()
        html = "<html><body><p>Hello World</p><script>console.log('test');</script></body></html>"
        text = scraper.extract_text(html)

        assert "Hello World" in text
        assert "console.log" not in text  # Script should be removed

    def test_find_links(self):
        """Test finding links with keyword."""
        scraper = WebScraper()
        html = """
        <html><body>
            <a href="/phosphogypsum-study">Phosphogypsum Research</a>
            <a href="/other-page">Other Page</a>
            <a href="/pg-analysis">PG Analysis</a>
        </body></html>
        """
        links = scraper.find_links(html, keyword="phosphogypsum")

        assert "/phosphogypsum-study" in links
        assert "/other-page" not in links


class TestDataStandardizer:
    """Tests for DataStandardizer functionality."""

    def test_init(self):
        """Test DataStandardizer initialization."""
        standardizer = DataStandardizer()
        assert standardizer.base_year == 2024
        assert standardizer.base_currency == "USD"

    def test_custom_init(self):
        """Test DataStandardizer with custom parameters."""
        standardizer = DataStandardizer(base_year=2023, base_currency="EUR")
        assert standardizer.base_year == 2023
        assert standardizer.base_currency == "EUR"

    def test_convert_unit_mass(self):
        """Test mass unit conversion."""
        standardizer = DataStandardizer()

        # Tonne to kg
        result = standardizer.convert_unit(1.0, "t", "kg")
        assert result == 1000.0

        # kg to g
        result = standardizer.convert_unit(1.0, "kg", "g")
        assert result == 1000.0

    def test_standardize_composition(self):
        """Test composition data standardization."""
        standardizer = DataStandardizer()

        composition = {"CaSO4": 0.92, "P2O5": 0.015, "Ra226": 500}

        result = standardizer.standardize_composition(
            composition, source="Test Paper", country="China"
        )

        assert len(result) > 0
        # Check that CaSO4 is in the results
        caso4_items = [r for r in result if r.parameter == "CaSO4"]
        assert len(caso4_items) == 1
        assert caso4_items[0].value == 0.92


class TestAPIConnector:
    """Tests for APIConnector functionality."""

    def test_init(self):
        """Test APIConnector initialization."""
        connector = APIConnector()
        assert connector.cache_dir.exists()
        assert "openalex" in connector.sources
        assert "unpaywall" in connector.sources
        assert "pubmed" in connector.sources

    def test_custom_cache_dir(self, tmp_path):
        """Test APIConnector with custom cache directory."""
        cache_dir = tmp_path / "api_cache"
        connector = APIConnector(cache_dir=cache_dir)
        assert connector.cache_dir == cache_dir
        assert cache_dir.exists()

    @patch("requests.get")
    def test_search_openalex_success(self, mock_get):
        """Test OpenAlex search with mocked response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"id": "W123", "title": "Phosphogypsum LCA Study"},
                {"id": "W456", "title": "REE Extraction from PG"},
            ]
        }
        mock_get.return_value = mock_response

        connector = APIConnector()
        results = connector.search_openalex("phosphogypsum LCA")

        assert len(results) == 2
        assert results[0]["title"] == "Phosphogypsum LCA Study"

    @patch("requests.get")
    def test_search_openalex_error(self, mock_get):
        """Test OpenAlex search error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        connector = APIConnector()

        with pytest.raises(Exception, match="OpenAlex API error"):
            connector.search_openalex("test query")


class TestModuleImports:
    """Tests for module imports and exports."""

    def test_imports_from_iodata(self):
        """Test that all components can be imported from iodata."""
        from pgloop.iodata import APIConnector, DataStandardizer, WebScraper

        assert PDFParser is not None
        assert WebScraper is not None
        assert DataStandardizer is not None
        assert APIConnector is not None

    def test_all_exports(self):
        """Test __all__ exports."""
        import pgloop.iodata as iodata

        expected_exports = ["PDFParser", "WebScraper", "DataStandardizer", "APIConnector"]
        for export in expected_exports:
            assert export in iodata.__all__


def main():
    print("Running Data Ingestion Tests...")
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    main()
