"""
Export Module

Handles exporting reports and dashboards to various formats.
"""

from typing import Any, Dict


class ReportExporter:
    """Exports structured reports to HTML, Excel, etc."""

    def to_html(self, data: Dict[str, Any], filepath: str, title: str = "Report"):
        """Export data to a simple HTML file."""
        html_content = f"<html><head><title>{title}</title></head><body>"
        html_content += f"<h1>{title}</h1>"
        for key, value in data.items():
            html_content += f"<h2>{key}</h2><pre>{value}</pre>"
        html_content += "</body></html>"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

    def to_excel(self, data: Dict[str, Any], filepath: str):
        """Export data to an Excel file using pandas."""
        try:
            import pandas as pd

            df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
            df.to_excel(filepath, index=False)
        except ImportError:
            raise ImportError("pandas is required to export to Excel.")
