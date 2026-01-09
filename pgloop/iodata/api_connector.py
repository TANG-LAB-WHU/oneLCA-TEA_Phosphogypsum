"""
API Connector Module

Connects to open-access databases and regulatory sources for LCI data.
"""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class APISource:
    """Configuration for an API data source."""
    
    name: str
    base_url: str
    api_type: str  # rest, graphql
    requires_auth: bool = False
    rate_limit: int = 100  # requests per minute


# Open access database configurations
OPEN_DATABASES = {
    "openalex": APISource(
        name="OpenAlex",
        base_url="https://api.openalex.org",
        api_type="rest",
        requires_auth=False,
        rate_limit=100
    ),
    "unpaywall": APISource(
        name="Unpaywall",
        base_url="https://api.unpaywall.org/v2",
        api_type="rest",
        requires_auth=False,  # Requires email
        rate_limit=100
    ),
    "pubmed": APISource(
        name="PubMed",
        base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        api_type="rest",
        requires_auth=False,
        rate_limit=10
    ),
}


class APIConnector:
    """
    Connects to open-access databases for literature and LCI data.
    
    Supported sources:
    - OpenAlex: Open scholarly metadata
    - Unpaywall: Open access article links
    - PubMed: Biomedical literature
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the API connector.
        
        Args:
            cache_dir: Directory for caching API responses
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not installed. Run: pip install requests")
        
        self.cache_dir = cache_dir or Path("./data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sources = OPEN_DATABASES.copy()
    
    def search_openalex(
        self,
        query: str,
        filters: Optional[Dict] = None,
        per_page: int = 25
    ) -> List[Dict]:
        """
        Search OpenAlex for scholarly works.
        
        Args:
            query: Search query
            filters: Additional filters (e.g., publication_year, concepts)
            per_page: Results per page
            
        Returns:
            List of work records
        """
        source = self.sources["openalex"]
        
        params = {
            "search": query,
            "per_page": per_page,
        }
        
        if filters:
            filter_parts = []
            for key, value in filters.items():
                filter_parts.append(f"{key}:{value}")
            params["filter"] = ",".join(filter_parts)
        
        response = requests.get(
            f"{source.base_url}/works",
            params=params,
            headers={"Accept": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("results", [])
        else:
            raise Exception(f"OpenAlex API error: {response.status_code}")
    
    def search_phosphogypsum_literature(
        self,
        keywords: List[str] = None,
        year_from: int = 2015,
        year_to: int = 2025
    ) -> List[Dict]:
        """
        Search for phosphogypsum-related literature.
        
        Args:
            keywords: Additional keywords
            year_from: Start year
            year_to: End year
            
        Returns:
            List of relevant publications
        """
        base_keywords = ["phosphogypsum"]
        if keywords:
            base_keywords.extend(keywords)
        
        query = " ".join(base_keywords)
        
        filters = {
            "publication_year": f"{year_from}-{year_to}",
            "type": "article",
        }
        
        return self.search_openalex(query, filters, per_page=50)
    
    def get_open_access_pdf(self, doi: str) -> Optional[str]:
        """
        Get open access PDF URL for a DOI using Unpaywall.
        
        Args:
            doi: Digital Object Identifier
            
        Returns:
            URL to open access PDF if available
        """
        source = self.sources["unpaywall"]
        
        # Unpaywall requires an email
        email = "researcher@example.com"  # Replace with actual email
        
        response = requests.get(
            f"{source.base_url}/{doi}",
            params={"email": email}
        )
        
        if response.status_code == 200:
            data = response.json()
            best_oa = data.get("best_oa_location")
            if best_oa:
                return best_oa.get("url_for_pdf")
        
        return None
    
    def fetch_elcd_data(self, process_name: str) -> Optional[Dict]:
        """
        Fetch data from ELCD (European Life Cycle Database).
        
        Note: ELCD data is typically accessed through downloaded datasets,
        not directly via API. This method reads from local ELCD cache.
        
        Args:
            process_name: Name of the process to look up
            
        Returns:
            Process data if found
        """
        elcd_cache = self.cache_dir / "elcd"
        elcd_file = elcd_cache / f"{process_name.lower().replace(' ', '_')}.json"
        
        if elcd_file.exists():
            with open(elcd_file, "r") as f:
                return json.load(f)
        
        return None
    
    def cache_response(
        self, 
        source: str, 
        query: str, 
        data: Any
    ) -> None:
        """Cache an API response for future use."""
        cache_file = self.cache_dir / source / f"{hash(query)}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_file, "w") as f:
            json.dump(data, f)
    
    def get_cached_response(
        self, 
        source: str, 
        query: str
    ) -> Optional[Any]:
        """Retrieve a cached API response."""
        cache_file = self.cache_dir / source / f"{hash(query)}.json"
        
        if cache_file.exists():
            with open(cache_file, "r") as f:
                return json.load(f)
        
        return None


if __name__ == "__main__":
    # Example usage
    connector = APIConnector()
    
    # Search for phosphogypsum LCA literature
    results = connector.search_phosphogypsum_literature(
        keywords=["LCA", "life cycle assessment"],
        year_from=2020
    )
    
    print(f"Found {len(results)} results")
    for work in results[:5]:
        print(f"- {work.get('title', 'No title')}")
