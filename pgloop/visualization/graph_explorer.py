"""
Knowledge Graph Explorer Module

Visualizes and explores the knowledge graph.
"""

import networkx as nx

try:
    from pyvis.network import Network

    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False


class KGExplorer:
    """Utilities for KG visualization."""

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    def export_html(self, output_path: str = "kg_visualization.html"):
        """Export interactive HTML visualization using PyVis."""
        if not PYVIS_AVAILABLE:
            print("PyVis not installed. Run: pip install pyvis")
            return

        nt = Network(
            height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False
        )
        nt.from_nx(self.graph)
        nt.show(output_path, local=False)
        print(f"KG visualization saved to {output_path}")

    def get_neighbors(self, node_id: str) -> list:
        """Get neighbors of a node."""
        if node_id in self.graph:
            return list(self.graph.neighbors(node_id))
        return []
