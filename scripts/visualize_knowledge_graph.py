
import json
import os
from pathlib import Path
import networkx as nx

def load_graph_with_viz_attributes(storage_path="data/processed/knowledge_graph"):
    """
    Load the knowledge graph from JSON files and add attributes for visualization.
    """
    nodes_path = Path(storage_path) / "kg_nodes.json"
    edges_path = Path(storage_path) / "kg_edges.json"
    
    if not nodes_path.exists() or not edges_path.exists():
        print(f"Error: Knowledge graph files not found in {storage_path}")
        return None

    G = nx.MultiDiGraph()
    
    # Load nodes
    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)
        for node in nodes:
            node_id = node.get("id")
            if node_id:
                props = {k: v for k, v in node.items() if k != "id"}
                
                # --- Visualization Attributes ---
                # Label
                if "name" in props:
                    props["label"] = str(props["name"])
                elif "title" in props:
                    props["label"] = str(props["title"])[:20] + "..."
                else:
                    props["label"] = node_id
                
                # Color mapping
                node_type = props.get("type", "Unknown")
                colors = {
                    "Country": "#FF9999",       # Red
                    "Composition": "#99FF99",   # Green
                    "Technology": "#9999FF",    # Blue
                    "Material": "#FFFF99",      # Yellow
                    "Source": "#CCCCCC",        # Gray
                    "Emission": "#FFCC99",      # Orange
                    "Cost": "#FF99FF"           # Pink
                }
                props["color"] = colors.get(node_type, "#FFFFFF")
                
                G.add_node(node_id, **props)
    
    # Load edges
    with open(edges_path, "r", encoding="utf-8") as f:
        edges = json.load(f)
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                props = {k: v for k, v in edge.items() if k not in ["source", "target"]}
                # Label for edge
                props["label"] = props.get("relation", "")
                G.add_edge(source, target, **props)
                
    return G

def print_graph_summary(G):
    """Print a summary of the graph contents."""
    if G is None: return

    print("\n" + "="*50)
    print("   PHOSPHOGYPSUM KNOWLEDGE GRAPH SUMMARY")
    print("="*50)
    print(f"Total Nodes: {G.number_of_nodes()}")
    print(f"Total Edges: {G.number_of_edges()}")
    
    # Count node types
    node_types = {}
    for _, data in G.nodes(data=True):
        ntype = data.get("type", "Unknown")
        node_types[ntype] = node_types.get(ntype, 0) + 1
        
    print("\n--- Node Types ---")
    for ntype, count in node_types.items():
        print(f"{ntype}: {count}")

    print("\n--- Sample Node Details ---")
    # Print details for a few nodes of each type
    for ntype in node_types:
        print(f"\nType: {ntype}")
        count = 0
        for node_id, data in G.nodes(data=True):
            if data.get("type") == ntype:
                # Format properties for readability (exclude internal viz props)
                props_str = ", ".join([f"{k}={v}" for k, v in data.items() 
                                     if k not in ['type', 'created_at', 'id', 'label', 'color', 'shape', 'font']])
                print(f"  - [{node_id}]: {props_str[:150]}...") 
                count += 1
                if count >= 3: break # Limit to 3 per type

def visualize_pyvis(G, output_path):
    """Visualize using Pyvis (Interactive HTML)."""
    try:
        from pyvis.network import Network
        print("\n[Visualization] Generating interactive graph with Pyvis...")
        
        # Initialize network
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", select_menu=True)
        net.from_nx(G)
        net.show_buttons(filter_=['physics'])
        
        output_abs = str(Path(output_path).resolve())
        net.save_graph(output_abs)
        print(f"  -> Saved: {output_path}")
        return True
    except ImportError:
        print("  -> Pyvis not installed. Skipping interactive graph.")
        return False
    except Exception as e:
        print(f"  -> Pyvis visualization failed: {e}")
        return False

def visualize_matplotlib(G, output_path):
    """Visualize using Matplotlib (Static Image)."""
    try:
        import matplotlib.pyplot as plt
        print("\n[Visualization] Generating static image with Matplotlib...")
        
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Nodes
        colors = [data.get("color", "#CCCCCC") for _, data in G.nodes(data=True)]
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=colors, alpha=0.9)
        
        # Labels
        labels = {n: data.get("label", n) for n, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        # Edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrowstyle='->', arrowsize=20)
        
        # Edge Labels (Relationship type)
        edge_labels = {}
        for u, v, k, d in G.edges(keys=True, data=True):
             edge_labels[(u, v)] = d.get("relation", "")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        
        plt.title("Phosphogypsum Knowledge Graph")
        plt.axis("off")
        
        output_image = str(Path(output_path).with_suffix(".png"))
        plt.savefig(output_image, dpi=300, bbox_inches="tight")
        print(f"  -> Saved: {output_image}")
        return True
    except ImportError:
        print("  -> Matplotlib not installed. Skipping static image.")
        return False
    except Exception as e:
        print(f"  -> Matplotlib visualization failed: {e}")
        return False

if __name__ == "__main__":
    # 1. Load Data
    kg = load_graph_with_viz_attributes()
    
    if kg:
        # 2. Print Text Summary
        print_graph_summary(kg)
        
        # 3. Generate Visualizations
        output_dir = Path("data/processed/knowledge_graph")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        visualize_pyvis(kg, output_dir / "kg_interactive.html")
        visualize_matplotlib(kg, output_dir / "kg_static.png")
