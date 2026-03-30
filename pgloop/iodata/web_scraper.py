"""
Web Scraper Module

Scrapes phosphogypsum related data from regulatory and news sites.
"""

from typing import List, Optional

import requests
from bs4 import BeautifulSoup


class WebScraper:
    """Scraper for public phosphogypsum data."""

    def __init__(self, user_agent: str = "Mozilla/5.0"):
        self.headers = {"User-Agent": user_agent}

    def scrape_url(self, url: str) -> Optional[str]:
        """Fetch content from a URL."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def extract_text(self, html: str) -> str:
        """Extract plain text from HTML."""
        soup = BeautifulSoup(html, "html.parser")
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        return soup.get_text(separator=" ", strip=True)

    def find_links(self, html: str, keyword: str = "phosphogypsum") -> List[str]:
        """Find links containing a keyword."""
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            if keyword.lower() in a.get_text().lower() or keyword.lower() in a["href"].lower():
                links.append(a["href"])
        return list(set(links))


def main():
    WebScraper()
    # Example: scrape EPA news (if allowed)
    # scraper = WebScraper()
    # text = scraper.scrape_url("https://www.epa.gov/radiation/phosphogypsum-stack-free-rules")
    pass


if __name__ == "__main__":
    main()
