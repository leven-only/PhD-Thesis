from pathlib import Path

import networkx as nx
import osmnx as ox
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

INPUT_GRAPHML_PATH = (
    BASE_DIR
    / "processed_network"
    / "step_02_with_ids"
    / "network_with_ids.graphml"
)

OUTPUT_DIR = (
    BASE_DIR
    / "processed_network"
    / "step_04_largest_component"
)


def get_total_length(G):
    total_length = 0.0

    for _, _, _, data in G.edges(keys=True, data=True):
        total_length += float(data.get("length", 0))

    return total_length


def summarize_component(G, component_nodes, component_id, component_rank):
    subgraph = G.subgraph(component_nodes)

    xs = [
        float(G.nodes[node]["x"])
        for node in component_nodes
        if "x" in G.nodes[node]
    ]

    ys = [
        float(G.nodes[node]["y"])
        for node in component_nodes
        if "y" in G.nodes[node]
    ]

    total_length = get_total_length(subgraph)

    return {
        "component_id": component_id,
        "component_rank_by_size": component_rank,
        "node_count": subgraph.number_of_nodes(),
        "edge_count": subgraph.number_of_edges(),
        "total_length_m": total_length,
        "centroid_x": sum(xs) / len(xs) if xs else None,
        "centroid_y": sum(ys) / len(ys) if ys else None,
        "min_x": min(xs) if xs else None,
        "max_x": max(xs) if xs else None,
        "min_y": min(ys) if ys else None,
        "max_y": max(ys) if ys else None,
    }


def keep_largest_weak_component(
    input_graphml_path=INPUT_GRAPHML_PATH,
    output_dir=OUTPUT_DIR,
):
    """
    Keep the largest weakly connected component and remove all smaller
    disconnected components.

    This step does not overwrite raw data or step_02 data.
    It creates a new processed graph containing only the main component.
    """

    input_graphml_path = Path(input_graphml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_graphml_path.exists():
        raise FileNotFoundError(f"Input GraphML file not found: {input_graphml_path}")

    G = ox.load_graphml(input_graphml_path)

    original_node_count = G.number_of_nodes()
    original_edge_count = G.number_of_edges()
    original_total_length = get_total_length(G)

    components = list(nx.weakly_connected_components(G))

    if not components:
        raise ValueError("No weakly connected components found.")

    component_items = []

    for index, component_nodes in enumerate(components, start=1):
        component_items.append(
            {
                "component_id": f"component_{index:04d}",
                "nodes": set(component_nodes),
                "node_count": len(component_nodes),
            }
        )

    component_items = sorted(
        component_items,
        key=lambda item: item["node_count"],
        reverse=True,
    )

    component_records = []

    for rank, item in enumerate(component_items, start=1):
        record = summarize_component(
            G=G,
            component_nodes=item["nodes"],
            component_id=item["component_id"],
            component_rank=rank,
        )

        if rank == 1:
            record["decision"] = "kept"
            record["reason"] = "largest weakly connected component"
        else:
            record["decision"] = "removed"
            record["reason"] = "disconnected from the main road network"

        component_records.append(record)

    largest_component_nodes = component_items[0]["nodes"]
    G_main = G.subgraph(largest_component_nodes).copy()

    new_node_count = G_main.number_of_nodes()
    new_edge_count = G_main.number_of_edges()
    new_total_length = get_total_length(G_main)

    removed_node_count = original_node_count - new_node_count
    removed_edge_count = original_edge_count - new_edge_count
    removed_total_length = original_total_length - new_total_length

    remaining_components = list(nx.weakly_connected_components(G_main))

    # Export processed graph
    ox.save_graphml(
        G_main,
        filepath=output_dir / "network_largest_component.graphml",
    )

    nodes, edges = ox.graph_to_gdfs(G_main)

    nodes.to_file(
        output_dir / "nodes_largest_component.geojson",
        driver="GeoJSON",
    )

    edges.to_file(
        output_dir / "edges_largest_component.geojson",
        driver="GeoJSON",
    )

    nodes.reset_index().to_csv(
        output_dir / "nodes_largest_component.csv",
        index=False,
        encoding="utf-8-sig",
    )

    edges.reset_index().to_csv(
        output_dir / "edges_largest_component.csv",
        index=False,
        encoding="utf-8-sig",
    )

    component_summary = pd.DataFrame(component_records)

    component_summary.to_csv(
        output_dir / "weak_component_decisions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    removed_components = component_summary[
        component_summary["decision"] == "removed"
    ]

    removed_components.to_csv(
        output_dir / "removed_components.csv",
        index=False,
        encoding="utf-8-sig",
    )

if __name__ == '__main__':
    keep_largest_weak_component()