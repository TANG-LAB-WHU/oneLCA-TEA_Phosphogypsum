"""
This script visualizes the knowledge graph.
"""

import json
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
                    "Country": "#FF9999",  # Red
                    "Composition": "#99FF99",  # Green
                    "Technology": "#9999FF",  # Blue
                    "Material": "#FFFF99",  # Yellow
                    "Source": "#CCCCCC",  # Gray
                    "Emission": "#FFCC99",  # Orange
                    "Cost": "#FF99FF",  # Pink
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
    if G is None:
        return

    print("\n" + "=" * 50)
    print("   PHOSPHOGYPSUM KNOWLEDGE GRAPH SUMMARY")
    print("=" * 50)
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
                props_str = ", ".join(
                    [
                        f"{k}={v}"
                        for k, v in data.items()
                        if k not in ["type", "created_at", "id", "label", "color", "shape", "font"]
                    ]
                )
                print(f"  - [{node_id}]: {props_str[:150]}...")
                count += 1
                if count >= 3:
                    break  # Limit to 3 per type


def visualize_pyvis(G, output_path):
    """Visualize using Pyvis (Interactive HTML)."""
    try:
        from pyvis.network import Network

        print("\n[Visualization] Generating interactive graph with Pyvis...")

        # Initialize network
        net = Network(
            height="750px", width="100%", bgcolor="#222222", font_color="white", select_menu=True
        )
        net.from_nx(G)
        net.show_buttons(filter_=["physics"])

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


def visualize_matplotlib(G, output_path, max_label_length=25):
    """
    Visualize using Matplotlib (Static Image) with short code labels.

    Features:
    - Short letter codes (C1, T1, S1, etc.) as node labels
    - Text table legend showing full name mappings
    - Larger figure size for better spacing
    - Kamada-Kawai layout for better node distribution
    - Legend for node types with color coding

    Args:
        G: NetworkX graph
        output_path: Path to save the image
        max_label_length: Maximum characters for legend text (default 25)
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        print("\n[Visualization] Generating static image with Matplotlib...")

        # Use larger figure for better spacing (with room for legend table)
        num_nodes = G.number_of_nodes()
        fig_size = max(14, min(24, num_nodes * 1.2))

        # Create figure with space for legend table at bottom
        fig = plt.figure(figsize=(fig_size, fig_size + 4))
        ax = fig.add_axes([0.05, 0.25, 0.9, 0.7])  # Leave space at bottom for legend

        # Use Kamada-Kawai layout for better distribution
        if num_nodes <= 50:
            try:
                pos = nx.kamada_kawai_layout(G, scale=2)
            except Exception:
                pos = nx.spring_layout(G, k=3.0, iterations=100, seed=42)
        else:
            pos = nx.spring_layout(G, k=5.0 / max(1, num_nodes**0.5), iterations=150, seed=42)

        # Color mapping by node type
        color_map = {
            "Country": "#FF9999",  # Red
            "Composition": "#99FF99",  # Green
            "Technology": "#9999FF",  # Blue
            "Material": "#FFFF99",  # Yellow
            "Source": "#CCCCCC",  # Gray
            "Emission": "#FFCC99",  # Orange
            "Cost": "#FF99FF",  # Pink
            "Impact": "#99FFFF",  # Cyan
            "Regulation": "#CC99FF",  # Purple
        }

        # Type abbreviations for short codes
        type_abbrev = {
            "Country": "Co",
            "Composition": "C",
            "Technology": "T",
            "Material": "M",
            "Source": "S",
            "Emission": "E",
            "Cost": "$",
            "Impact": "I",
            "Regulation": "R",
        }

        # Generate short codes and build legend mapping
        short_labels = {}
        legend_entries = []  # List of (code, type, full_name)
        type_counters = {}

        for node, data in G.nodes(data=True):
            ntype = data.get("type", "Unknown")
            abbrev = type_abbrev.get(ntype, "?")

            # Increment counter for this type
            type_counters[ntype] = type_counters.get(ntype, 0) + 1
            count = type_counters[ntype]

            # Create short code like C1, T1, S1
            short_code = f"{abbrev}{count}"
            short_labels[node] = short_code

            # Get full name for legend
            full_name = data.get("label", data.get("name", node))
            if len(str(full_name)) > 40:
                full_name = str(full_name)[:37] + "..."

            legend_entries.append((short_code, ntype, full_name, color_map.get(ntype, "#FFFFFF")))

        # Get node colors
        colors = []
        node_types_in_graph = set()
        for _, data in G.nodes(data=True):
            ntype = data.get("type", "Unknown")
            node_types_in_graph.add(ntype)
            colors.append(color_map.get(ntype, "#FFFFFF"))

        # Draw nodes with size based on degree
        node_sizes = [max(1000, 1800 + G.degree(n) * 100) for n in G.nodes()]
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color=colors,
            alpha=0.85,
            ax=ax,
            edgecolors="#333333",
            linewidths=2,
        )

        # Draw short code labels directly on nodes
        nx.draw_networkx_labels(
            G,
            pos,
            labels=short_labels,
            font_size=12,
            font_weight="bold",
            ax=ax,
            font_color="#222222",
        )

        # Draw edges with curved arrows
        nx.draw_networkx_edges(
            G,
            pos,
            width=1.0,
            alpha=0.5,
            arrowstyle="-|>",
            arrowsize=18,
            connectionstyle="arc3,rad=0.1",
            edge_color="#555555",
            ax=ax,
        )

        # Edge relation abbreviations
        edge_abbrev = {
            "references": "R",
            "produces": "P",
            "treated_by": "Tr",
            "requires": "Rq",
            "emits": "Em",
            "causes": "Ca",
            "costs": "Co",
            "regulated_by": "Rg",
            "located_in": "L",
        }

        # Edge Labels - only draw if not too many, use short codes
        if G.number_of_edges() <= 30:
            edge_labels = {}
            for u, v, k, d in G.edges(keys=True, data=True):
                relation = d.get("relation", "")
                edge_labels[(u, v)] = edge_abbrev.get(relation, relation[:2] if relation else "")
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, font_size=8, alpha=0.9, ax=ax
            )

        # Add color legend for node types (top left)
        legend_handles = []
        for ntype in sorted(node_types_in_graph):
            color = color_map.get(ntype, "#FFFFFF")
            abbrev = type_abbrev.get(ntype, "?")
            patch = mpatches.Patch(color=color, label=f"{abbrev} = {ntype}", alpha=0.85)
            legend_handles.append(patch)

        ax.legend(
            handles=legend_handles,
            loc="upper left",
            fontsize=10,
            title="Node Types",
            title_fontsize=11,
            framealpha=0.95,
        )

        plt.title("Phosphogypsum Knowledge Graph", fontsize=18, fontweight="bold", pad=15)
        ax.axis("off")

        # Create text table legend at bottom showing code -> full name mappings
        ax_table = fig.add_axes([0.05, 0.02, 0.9, 0.2])
        ax_table.axis("off")

        # Build table text
        table_header = "Code  │  Type          │  Full Name"
        table_separator = "──────┼────────────────┼" + "─" * 45
        table_lines = [table_header, table_separator]

        for code, ntype, full_name, color in sorted(legend_entries, key=lambda x: (x[1], x[0])):
            line = f"{code:5} │  {ntype:14} │  {full_name}"
            table_lines.append(line)

        table_text = "\n".join(table_lines)
        ax_table.text(
            0.0,
            1.0,
            table_text,
            fontsize=9,
            fontfamily="monospace",
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="#f8f8f8", edgecolor="#cccccc", alpha=0.95
            ),
        )

        # Save figure
        output_image = str(Path(output_path).with_suffix(".png"))
        plt.savefig(output_image, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)

        print(f"  -> Saved: {output_image}")
        print(
            f"     Figure size: {fig_size}x{fig_size + 4}, "
            f"Nodes: {num_nodes}, Edges: {G.number_of_edges()}"
        )
        return True

    except ImportError:
        print("  -> Matplotlib not installed. Skipping static image.")
        return False
    except Exception as e:
        import traceback

        print(f"  -> Matplotlib visualization failed: {e}")
        traceback.print_exc()
        return False


def main():
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


if __name__ == "__main__":
    main()
