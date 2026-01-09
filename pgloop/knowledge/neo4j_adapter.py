"""
Neo4j Knowledge Graph Adapter

Provides Neo4j database connectivity for the Phosphogypsum Knowledge Graph.
Enables production-scale graph storage with Cypher query support.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""
    
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"


class Neo4jAdapter:
    """
    Neo4j adapter for PhosphogypsumKG.
    
    Provides:
    - Connection management
    - CRUD operations for nodes and edges
    - Cypher query execution
    - Sync with NetworkX graph
    """
    
    def __init__(self, config: Neo4jConfig = None):
        """
        Initialize Neo4j adapter.
        
        Args:
            config: Neo4j connection configuration
        """
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "Neo4j driver not installed. Run: pip install neo4j"
            )
        
        self.config = config or Neo4jConfig()
        self._driver = None
    
    def connect(self) -> bool:
        """
        Establish connection to Neo4j.
        
        Returns:
            True if connection successful
        """
        try:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password)
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.config.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def close(self) -> None:
        """Close the database connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # ==================== Node Operations ====================
    
    def create_node(
        self,
        node_id: str,
        node_type: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        """
        Create a node in Neo4j.
        
        Args:
            node_id: Unique node identifier
            node_type: Node label (e.g., Country, Technology)
            properties: Node properties
            
        Returns:
            True if successful
        """
        props = properties or {}
        props["node_id"] = node_id
        
        query = f"""
        MERGE (n:{node_type} {{node_id: $node_id}})
        SET n += $properties
        RETURN n
        """
        
        try:
            with self._driver.session(database=self.config.database) as session:
                session.run(query, node_id=node_id, properties=props)
            return True
        except Exception as e:
            logger.error(f"Failed to create node {node_id}: {e}")
            return False
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get a node by ID."""
        query = """
        MATCH (n {node_id: $node_id})
        RETURN n, labels(n) as labels
        """
        
        try:
            with self._driver.session(database=self.config.database) as session:
                result = session.run(query, node_id=node_id)
                record = result.single()
                if record:
                    node = dict(record["n"])
                    node["_labels"] = record["labels"]
                    return node
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
        return None
    
    def get_nodes_by_label(self, label: str) -> List[Dict]:
        """Get all nodes with a specific label."""
        query = f"""
        MATCH (n:{label})
        RETURN n
        """
        
        nodes = []
        try:
            with self._driver.session(database=self.config.database) as session:
                result = session.run(query)
                for record in result:
                    nodes.append(dict(record["n"]))
        except Exception as e:
            logger.error(f"Failed to get nodes by label {label}: {e}")
        return nodes
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its relationships."""
        query = """
        MATCH (n {node_id: $node_id})
        DETACH DELETE n
        """
        
        try:
            with self._driver.session(database=self.config.database) as session:
                session.run(query, node_id=node_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete node {node_id}: {e}")
            return False
    
    # ==================== Edge Operations ====================
    
    def create_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        """
        Create an edge between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation: Relationship type (uppercase in Neo4j)
            properties: Edge properties
        """
        props = properties or {}
        rel_type = relation.upper()
        
        query = f"""
        MATCH (a {{node_id: $source_id}})
        MATCH (b {{node_id: $target_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $properties
        RETURN r
        """
        
        try:
            with self._driver.session(database=self.config.database) as session:
                session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id,
                    properties=props
                )
            return True
        except Exception as e:
            logger.error(f"Failed to create edge {source_id}->{target_id}: {e}")
            return False
    
    def get_edges(self, node_id: str, direction: str = "out") -> List[Dict]:
        """
        Get edges connected to a node.
        
        Args:
            node_id: Node ID
            direction: "out", "in", or "both"
        """
        if direction == "out":
            query = """
            MATCH (n {node_id: $node_id})-[r]->(m)
            RETURN type(r) as relation, r, m.node_id as target_id, m
            """
        elif direction == "in":
            query = """
            MATCH (n {node_id: $node_id})<-[r]-(m)
            RETURN type(r) as relation, r, m.node_id as source_id, m
            """
        else:
            query = """
            MATCH (n {node_id: $node_id})-[r]-(m)
            RETURN type(r) as relation, r, m.node_id as other_id, m
            """
        
        edges = []
        try:
            with self._driver.session(database=self.config.database) as session:
                result = session.run(query, node_id=node_id)
                for record in result:
                    edges.append({
                        "relation": record["relation"],
                        "properties": dict(record["r"]),
                        "connected_node": dict(record["m"]),
                    })
        except Exception as e:
            logger.error(f"Failed to get edges for {node_id}: {e}")
        return edges
    
    # ==================== Query Operations ====================
    
    def run_cypher(
        self,
        query: str,
        parameters: Dict[str, Any] = None
    ) -> List[Dict]:
        """
        Execute a custom Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dicts
        """
        params = parameters or {}
        results = []
        
        try:
            with self._driver.session(database=self.config.database) as session:
                result = session.run(query, **params)
                for record in result:
                    results.append(dict(record))
        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
        
        return results
    
    def get_technology_lci(self, tech_code: str) -> Dict:
        """
        Get complete LCI data for a technology using Cypher.
        
        Returns:
            Dict with inputs, outputs, emissions
        """
        tech_id = f"tech_{tech_code.lower()}"
        
        query = """
        MATCH (t:Technology {node_id: $tech_id})
        OPTIONAL MATCH (t)-[:REQUIRES]->(input:Material)
        OPTIONAL MATCH (t)-[:EMITS]->(emission:Emission)
        RETURN t,
               collect(DISTINCT input) as inputs,
               collect(DISTINCT emission) as emissions
        """
        
        try:
            with self._driver.session(database=self.config.database) as session:
                result = session.run(query, tech_id=tech_id)
                record = result.single()
                
                if record:
                    return {
                        "technology": dict(record["t"]) if record["t"] else {},
                        "inputs": [dict(n) for n in record["inputs"] if n],
                        "emissions": [dict(n) for n in record["emissions"] if n],
                    }
        except Exception as e:
            logger.error(f"Failed to get LCI for {tech_code}: {e}")
        
        return {"technology": {}, "inputs": [], "emissions": []}
    
    def find_shortest_path(
        self,
        start_id: str,
        end_id: str,
        max_hops: int = 5
    ) -> List[str]:
        """Find shortest path between two nodes."""
        query = f"""
        MATCH path = shortestPath(
            (a {{node_id: $start_id}})-[*1..{max_hops}]-(b {{node_id: $end_id}})
        )
        RETURN [n IN nodes(path) | n.node_id] as path_nodes
        """
        
        try:
            with self._driver.session(database=self.config.database) as session:
                result = session.run(query, start_id=start_id, end_id=end_id)
                record = result.single()
                if record:
                    return record["path_nodes"]
        except Exception as e:
            logger.error(f"Path search failed: {e}")
        
        return []
    
    # ==================== Sync with NetworkX ====================
    
    def import_from_networkx(self, nx_graph) -> int:
        """
        Import nodes and edges from a NetworkX graph.
        
        Args:
            nx_graph: NetworkX graph object
            
        Returns:
            Number of items imported
        """
        count = 0
        
        # Import nodes
        for node_id, props in nx_graph.nodes(data=True):
            node_type = props.get("type", "Node")
            if self.create_node(node_id, node_type, dict(props)):
                count += 1
        
        # Import edges
        for source, target, props in nx_graph.edges(data=True):
            relation = props.get("relation", "RELATED_TO")
            if self.create_edge(source, target, relation, dict(props)):
                count += 1
        
        logger.info(f"Imported {count} items from NetworkX")
        return count
    
    def export_to_networkx(self):
        """
        Export Neo4j graph to NetworkX.
        
        Returns:
            NetworkX MultiDiGraph
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX not installed")
        
        G = nx.MultiDiGraph()
        
        # Export nodes
        query = "MATCH (n) RETURN n, labels(n) as labels"
        with self._driver.session(database=self.config.database) as session:
            result = session.run(query)
            for record in result:
                props = dict(record["n"])
                node_id = props.pop("node_id", None)
                if node_id:
                    props["type"] = record["labels"][0] if record["labels"] else "Node"
                    G.add_node(node_id, **props)
        
        # Export edges
        query = "MATCH (a)-[r]->(b) RETURN a.node_id as source, b.node_id as target, type(r) as relation, r"
        with self._driver.session(database=self.config.database) as session:
            result = session.run(query)
            for record in result:
                props = dict(record["r"])
                props["relation"] = record["relation"].lower()
                G.add_edge(record["source"], record["target"], **props)
        
        return G
    
    # ==================== Utilities ====================
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        stats_query = """
        MATCH (n)
        WITH labels(n) as labels, count(n) as count
        UNWIND labels as label
        RETURN label, sum(count) as node_count
        """
        
        edge_query = """
        MATCH ()-[r]->()
        RETURN type(r) as relation, count(r) as edge_count
        """
        
        stats = {"nodes_by_type": {}, "edges_by_relation": {}}
        
        try:
            with self._driver.session(database=self.config.database) as session:
                # Node stats
                for record in session.run(stats_query):
                    stats["nodes_by_type"][record["label"]] = record["node_count"]
                
                # Edge stats
                for record in session.run(edge_query):
                    stats["edges_by_relation"][record["relation"]] = record["edge_count"]
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
        
        stats["total_nodes"] = sum(stats["nodes_by_type"].values())
        stats["total_edges"] = sum(stats["edges_by_relation"].values())
        
        return stats
    
    def clear_database(self) -> bool:
        """Clear all nodes and relationships. Use with caution!"""
        query = "MATCH (n) DETACH DELETE n"
        
        try:
            with self._driver.session(database=self.config.database) as session:
                session.run(query)
            logger.warning("Database cleared!")
            return True
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False
