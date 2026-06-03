"""
Builds and queries a knowledge graph for phosphogypsum LCA-TEA data.
Uses NetworkX for lightweight local graphs, with optional Neo4j support.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


# Node type definitions
NODE_TYPES = [
    "Country",
    "Composition",
    "Technology",
    "Material",
    "Emission",
    "Impact",
    "Cost",
    "Source",
    "Regulation",
]

# Edge type definitions
EDGE_TYPES = [
    "produces",
    "treated_by",
    "requires",
    "emits",
    "causes",
    "costs",
    "references",
    "regulated_by",
    "located_in",
]


@dataclass
class KGNode:
    """Represents a node in the knowledge graph."""

    id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class KGEdge:
    """Represents an edge in the knowledge graph."""

    source_id: str
    target_id: str
    relation: str
    properties: Dict[str, Any] = field(default_factory=dict)


class PhosphogypsumKG:
    """
    Knowledge Graph for phosphogypsum LCA-TEA data.

    Node Types:
    - Country: Phosphogypsum producing countries
    - Composition: PG chemical composition data
    - Technology: Treatment technologies
    - Material: Input/output materials
    - Emission: Environmental emissions
    - Impact: Impact assessment results
    - Cost: Cost data
    - Source: Literature/data sources
    - Regulation: Regulatory limits

    Edge Types:
    - produces: Country -> Composition
    - treated_by: Composition -> Technology
    - requires: Technology -> Material
    - emits: Technology -> Emission
    - causes: Emission -> Impact
    - costs: Technology -> Cost
    - references: * -> Source
    - regulated_by: Country -> Regulation
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the knowledge graph.

        Args:
            storage_path: Path to store/load the graph
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX not installed. Run: pip install networkx")

        self.graph = nx.MultiDiGraph()
        self.storage_path = storage_path or Path("./data/processed/kg")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Try to load existing graph
        self._load_graph()

    # ==================== Node Operations ====================

    def add_node(self, node_id: str, node_type: str, properties: Dict[str, Any] = None) -> str:
        """
        Add a node to the knowledge graph.

        Args:
            node_id: Unique identifier for the node
            node_type: Type of node (must be in NODE_TYPES)
            properties: Additional properties

        Returns:
            The node ID
        """
        if node_type not in NODE_TYPES:
            raise ValueError(f"Invalid node type: {node_type}. Must be one of {NODE_TYPES}")

        props = properties or {}
        props["type"] = node_type
        props["created_at"] = datetime.now().isoformat()

        self.graph.add_node(node_id, **props)
        return node_id

    def add_country(
        self,
        name: str,
        region: str,
        production_mt: float = None,
        utilization_rate: float = None,
        **kwargs,
    ) -> str:
        """Add a phosphogypsum producing country."""
        properties = {
            "name": name,
            "region": region,
            "production_mt": production_mt,
            "utilization_rate": utilization_rate,
            **kwargs,
        }
        country_name = str(name or "unknown").lower()
        return self.add_node(f"country_{country_name}", "Country", properties)

    def add_composition(
        self,
        name: str,
        country: str,
        CaSO4: float = None,
        P2O5: float = None,
        F: float = None,
        Ra226: float = None,
        **kwargs,
    ) -> str:
        """Add phosphogypsum composition data."""
        properties = {
            "name": name,
            "country": country,
            "CaSO4": CaSO4,
            "P2O5": P2O5,
            "F": F,
            "Ra226_Bq_kg": Ra226,
            **kwargs,
        }
        node_id = self.add_node(f"comp_{str(name or 'unknown').lower()}", "Composition", properties)

        # Link to country
        country_id = f"country_{str(country or 'unknown').lower()}"
        if self.graph.has_node(country_id):
            self.add_edge(country_id, node_id, "produces")

        return node_id

    def add_technology(
        self, name: str, code: str, trl: int = None, capacity_t_year: float = None, **kwargs
    ) -> str:
        """Add a treatment technology."""
        properties = {
            "name": name,
            "code": code,
            "trl": trl,
            "capacity_t_year": capacity_t_year,
            **kwargs,
        }
        return self.add_node(f"tech_{str(code or 'unknown').lower()}", "Technology", properties)

    def add_material(
        self,
        name: str,
        flow_type: str,  # input, output
        quantity: float = None,
        unit: str = "kg",
        **kwargs,
    ) -> str:
        """Add a material flow."""
        properties = {
            "name": name,
            "flow_type": flow_type,
            "quantity": quantity,
            "unit": unit,
            **kwargs,
        }
        mat_name = str(name or "unknown").lower().replace(" ", "_")
        return self.add_node(f"mat_{mat_name}", "Material", properties)

    def add_source(
        self, doi: str = None, title: str = None, authors: str = None, year: int = None, **kwargs
    ) -> str:
        """Add a literature source."""
        properties = {"doi": doi, "title": title, "authors": authors, "year": year, **kwargs}
        if doi:
            source_id = doi.replace("/", "_").replace(".", "_")
        elif title:
            source_id = title[:20].lower()
        else:
            source_id = "unknown"

        return self.add_node(f"src_{source_id}", "Source", properties)

    # ==================== Edge Operations ====================

    def add_edge(
        self, source_id: str, target_id: str, relation: str, properties: Dict[str, Any] = None
    ) -> None:
        """
        Add an edge to the knowledge graph.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation: Relation type (must be in EDGE_TYPES)
            properties: Additional edge properties
        """
        if relation not in EDGE_TYPES:
            raise ValueError(f"Invalid relation: {relation}. Must be one of {EDGE_TYPES}")

        props = properties or {}
        props["relation"] = relation

        self.graph.add_edge(source_id, target_id, **props)

    def link_technology_inputs(self, tech_id: str, material_ids: List[str]) -> None:
        """Link input materials to a technology."""
        for mat_id in material_ids:
            self.add_edge(tech_id, mat_id, "requires")

    def link_technology_emissions(self, tech_id: str, emission_ids: List[str]) -> None:
        """Link emissions to a technology."""
        for em_id in emission_ids:
            self.add_edge(tech_id, em_id, "emits")

    def add_source_reference(self, node_id: str, source_id: str) -> None:
        """Link a node to its source."""
        self.add_edge(node_id, source_id, "references")

    # ==================== Query Operations ====================

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get a node by ID."""
        if self.graph.has_node(node_id):
            return dict(self.graph.nodes[node_id])
        return None

    def get_nodes_by_type(self, node_type: str) -> List[Dict]:
        """Get all nodes of a specific type."""
        nodes = []
        for node_id, props in self.graph.nodes(data=True):
            if props.get("type") == node_type:
                nodes.append({"id": node_id, **props})
        return nodes

    def get_country_compositions(self, country: str) -> List[Dict]:
        """Get all compositions for a country."""
        country_id = f"country_{country.lower()}"
        compositions = []

        if self.graph.has_node(country_id):
            for _, target, data in self.graph.out_edges(country_id, data=True):
                if data.get("relation") == "produces":
                    comp = self.get_node(target)
                    if comp:
                        compositions.append({"id": target, **comp})

        return compositions

    def get_technology_lci(self, tech_code: str) -> Dict:
        """
        Get complete LCI data for a technology.

        Returns:
            Dict with inputs, outputs, emissions
        """
        tech_id = f"tech_{tech_code.lower()}"
        lci = {"inputs": [], "outputs": [], "emissions": []}

        if not self.graph.has_node(tech_id):
            return lci

        for _, target, data in self.graph.out_edges(tech_id, data=True):
            relation = data.get("relation")
            node = self.get_node(target)

            if relation == "requires":
                if node and node.get("flow_type") == "input":
                    lci["inputs"].append(node)
            elif relation == "emits":
                lci["emissions"].append(node)

        return lci

    def find_data_gaps(self) -> Dict[str, List[str]]:
        """
        Identify nodes missing critical data.

        Returns:
            Dict mapping node types to list of incomplete node IDs
        """
        gaps = {}

        # Required properties by node type
        required = {
            "Composition": ["CaSO4", "P2O5", "Ra226_Bq_kg"],
            "Technology": ["trl", "capacity_t_year"],
            "Material": ["quantity", "unit"],
        }

        for node_id, props in self.graph.nodes(data=True):
            node_type = props.get("type")
            if node_type in required:
                missing = [p for p in required[node_type] if props.get(p) is None]
                if missing:
                    if node_type not in gaps:
                        gaps[node_type] = []
                    gaps[node_type].append({"id": node_id, "missing": missing})

        return gaps

    # ==================== Persistence ====================

    def save_graph(self) -> None:
        """Save the graph to disk."""
        nodes_file = self.storage_path / "kg_nodes.json"
        edges_file = self.storage_path / "kg_edges.json"

        # Save nodes
        nodes = []
        for node_id, props in self.graph.nodes(data=True):
            nodes.append({"id": node_id, **props})

        with open(nodes_file, "w", encoding="utf-8") as f:
            json.dump(nodes, f, indent=2, default=str)

        # Save edges
        edges = []
        for source, target, props in self.graph.edges(data=True):
            edges.append({"source": source, "target": target, **props})

        with open(edges_file, "w", encoding="utf-8") as f:
            json.dump(edges, f, indent=2, default=str)

    def _load_graph(self) -> None:
        """Load the graph from disk if it exists."""
        nodes_file = self.storage_path / "kg_nodes.json"
        edges_file = self.storage_path / "kg_edges.json"

        if nodes_file.exists():
            with open(nodes_file, "r", encoding="utf-8") as f:
                nodes = json.load(f)
                for node in nodes:
                    node_id = node.pop("id")
                    self.graph.add_node(node_id, **node)

        if edges_file.exists():
            with open(edges_file, "r", encoding="utf-8") as f:
                edges = json.load(f)
                for edge in edges:
                    source = edge.pop("source")
                    target = edge.pop("target")
                    self.graph.add_edge(source, target, **edge)

    def get_statistics(self) -> Dict:
        """Get graph statistics."""
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "nodes_by_type": {},
            "edges_by_relation": {},
        }

        for _, props in self.graph.nodes(data=True):
            node_type = props.get("type", "unknown")
            stats["nodes_by_type"][node_type] = stats["nodes_by_type"].get(node_type, 0) + 1

        for _, _, props in self.graph.edges(data=True):
            relation = props.get("relation", "unknown")
            stats["edges_by_relation"][relation] = stats["edges_by_relation"].get(relation, 0) + 1

        return stats


def main():
    # Example usage
    kg = PhosphogypsumKG()

    # Add sample data
    kg.add_country("China", "Asia", production_mt=81, utilization_rate=0.45)
    kg.add_country("Morocco", "Africa", production_mt=30, utilization_rate=0.2)
    kg.add_country("USA", "North America", production_mt=25, utilization_rate=0.05)

    kg.add_composition("China_typical", "China", CaSO4=0.92, P2O5=0.01, F=0.006, Ra226=500)

    kg.add_technology("Cement Additive", "PG-CM", trl=9, capacity_t_year=100000)
    kg.add_technology("Stack Disposal", "PG-SD", trl=9, capacity_t_year=1000000)

    # Save and print stats
    kg.save_graph()
    print(kg.get_statistics())


if __name__ == "__main__":
    main()
