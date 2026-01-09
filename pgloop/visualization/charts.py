"""Plotting utilities for LCA and TEA results."""

from typing import Any, Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class LCAPlots:
    """LCA visualization utilities."""
    
    @staticmethod
    def impact_comparison_bar(
        results: Dict[str, Dict[str, float]],
        categories: List[str] = None,
        title: str = "Impact Comparison"
    ):
        """Create bar chart comparing impacts across pathways."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib not installed")
        
        pathways = list(results.keys())
        categories = categories or list(results[pathways[0]].keys())
        
        x = np.arange(len(categories))
        width = 0.8 / len(pathways)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, pathway in enumerate(pathways):
            values = [results[pathway].get(cat, 0) for cat in categories]
            ax.bar(x + i * width, values, width, label=pathway)
        
        ax.set_xlabel("Impact Category")
        ax.set_ylabel("Impact Value")
        ax.set_title(title)
        ax.set_xticks(x + width * (len(pathways) - 1) / 2)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def contribution_pie(
        contributions: Dict[str, float],
        title: str = "Impact Contribution"
    ):
        """Create pie chart for contribution analysis."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib not installed")
        
        labels = list(contributions.keys())
        values = list(contributions.values())
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title(title)
        
        return fig


class TEAPlots:
    """TEA visualization utilities."""
    
    @staticmethod
    def cost_breakdown_bar(
        costs: Dict[str, Dict[str, float]],
        title: str = "Cost Breakdown"
    ):
        """Stacked bar chart for cost breakdown."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib not installed")
        
        pathways = list(costs.keys())
        categories = list(costs[pathways[0]].keys())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bottom = np.zeros(len(pathways))
        for cat in categories:
            values = [costs[p].get(cat, 0) for p in pathways]
            ax.bar(pathways, values, bottom=bottom, label=cat)
            bottom += np.array(values)
        
        ax.set_xlabel("Pathway")
        ax.set_ylabel("Cost (USD/tonne)")
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def clcc_slcc_comparison(
        results: Dict[str, tuple],
        title: str = "CLCC vs SLCC Comparison"
    ):
        """Compare CLCC and SLCC across pathways."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib not installed")
        
        pathways = list(results.keys())
        clcc = [results[p][0] for p in pathways]
        slcc = [results[p][1] for p in pathways]
        
        x = np.arange(len(pathways))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, clcc, width, label="CLCC")
        ax.bar(x + width/2, slcc, width, label="SLCC")
        
        ax.set_xlabel("Pathway")
        ax.set_ylabel("Cost (USD/tonne)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(pathways)
        ax.legend()
        
        plt.tight_layout()
        return fig


class ReportExporter:
    """Export results to various formats."""
    
    @staticmethod
    def to_excel(results: Dict, filepath: str):
        """Export results to Excel."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas not installed")
        
        with pd.ExcelWriter(filepath) as writer:
            for sheet_name, data in results.items():
                if isinstance(data, dict):
                    df = pd.DataFrame([data])
                elif isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name=sheet_name[:31])
    
    @staticmethod
    def to_html(results: Dict, filepath: str, title: str = "PG-LCA-TEA Results"):
        """Export results to HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
        """
        
        for section, data in results.items():
            html += f"<h2>{section}</h2>"
            if isinstance(data, dict):
                html += "<table>"
                for k, v in data.items():
                    html += f"<tr><td>{k}</td><td>{v}</td></tr>"
                html += "</table>"
        
        html += "</body></html>"
        
        with open(filepath, "w") as f:
            f.write(html)
