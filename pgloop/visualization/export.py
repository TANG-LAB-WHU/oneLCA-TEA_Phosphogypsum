"""
Export Module

Handles exporting reports and dashboards to various formats.
"""

from typing import Any, Dict


class ReportExporter:
    """Exports structured reports to HTML, Excel, etc."""

    @staticmethod
    def to_excel(results: Dict[str, Any], filepath: str):
        """Export results to Excel."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to export to Excel.")

        with pd.ExcelWriter(filepath) as writer:
            for sheet_name, data in results.items():
                if isinstance(data, dict):
                    df = pd.DataFrame([data])
                elif isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([{"value": data}])
                df.to_excel(writer, sheet_name=str(sheet_name)[:31])

    @staticmethod
    def to_html(results: Dict[str, Any], filepath: str, title: str = "PG-LCA-TEA Results"):
        """Export results to HTML report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
"""
        for section, data in results.items():
            html += f"<h2>{section}</h2>\n"
            if isinstance(data, dict):
                html += "<table>\n"
                for k, v in data.items():
                    html += f"<tr><td>{k}</td><td>{v}</td></tr>\n"
                html += "</table>\n"
            elif isinstance(data, list):
                html += "<table>\n"
                for item in data:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            html += f"<tr><td>{k}</td><td>{v}</td></tr>\n"
                    else:
                        html += f"<tr><td>{item}</td></tr>\n"
                html += "</table>\n"
            else:
                html += f"<pre>{data}</pre>\n"

        html += "</body>\n</html>\n"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
