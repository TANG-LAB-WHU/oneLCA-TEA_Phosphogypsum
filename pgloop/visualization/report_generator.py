"""
Report Generator Module

Generates structured reports in PDF/Excel/HTML formats.
"""

from typing import Dict, Any, List
from pgloop.visualization.export import ReportExporter


class ReportGenerator:
    """High-level report generator for LCA-TEA results."""
    
    def __init__(self, project_name: str = "PG-LCA-TEA"):
        self.project_name = project_name
        self.exporter = ReportExporter()
    
    def generate_summary_report(self, results: Dict[str, Any], output_path: str):
        """Generate a complete summary report."""
        if output_path.endswith(".html"):
            self.exporter.to_html(results, output_path, title=f"{self.project_name} Summary Report")
        elif output_path.endswith(".xlsx"):
            self.exporter.to_excel(results, output_path)
        else:
            # Default to text or throw error
            with open(output_path, "w") as f:
                f.write(str(results))
