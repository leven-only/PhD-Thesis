from pathlib import Path

import osmnx as ox
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

STEP_04_GRAPHML_PATH = (
    BASE_DIR
    / "processed_network"
    / "step_04_largest_component"
    / "network_largest_component.graphml"
)

OUTPUT_DIR = (
    BASE_DIR
    / "processed_network"
    / "step_05_dead_end_diagnosis"
)


def count_dead_end_nodes(
    graphml_path=STEP_04_GRAPHML_PATH,
):
    G = ox.load_graphml(Path(graphml_path))

    dead_end_nodes = []

    for node in G.nodes:
        neighbors = set(G.predecessors(node)) | set(G.successors(node))
        neighbors.discard(node)

        if len(neighbors) == 1 or 0:
            dead_end_nodes.append(node)

    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Dead-end nodes: {len(dead_end_nodes)}")

    return dead_end_nodes

def get_unique_neighbors(G, node):
    neighbors = set(G.predecessors(node)) | set(G.successors(node))
    neighbors.discard(node)
    return neighbors


def get_incident_edges(G, node):
    records = []

    for u, v, key, data in G.out_edges(node, keys=True, data=True):
        records.append((u, v, key, data, "out"))

    for u, v, key, data in G.in_edges(node, keys=True, data=True):
        records.append((u, v, key, data, "in"))

    return records


def stringify_unique(values):
    clean_values = [
        str(value)
        for value in values
        if value is not None and str(value) != "nan" and str(value) != ""
    ]

    return "|".join(sorted(set(clean_values)))


def export_dead_end_nodes(
    graphml_path=STEP_04_GRAPHML_PATH,
    output_dir=OUTPUT_DIR,
):
    graphml_path = Path(graphml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    G = ox.load_graphml(graphml_path)

    records = []

    for node, node_data in G.nodes(data=True):
        unique_neighbors = get_unique_neighbors(G, node)

        if len(unique_neighbors) != 1:
            continue

        incident_edges = get_incident_edges(G, node)

        edge_ids = []
        highways = []
        names = []
        lengths = []
        oneways = []
        bridges = []
        tunnels = []

        for _, _, _, edge_data, _ in incident_edges:
            edge_ids.append(edge_data.get("road_edge_id"))
            highways.append(edge_data.get("highway"))
            names.append(edge_data.get("name"))
            oneways.append(edge_data.get("oneway"))
            bridges.append(edge_data.get("bridge"))
            tunnels.append(edge_data.get("tunnel"))

            try:
                lengths.append(float(edge_data.get("length", 0)))
            except (TypeError, ValueError):
                pass

        records.append(
            {
                "node": str(node),
                "road_node_id": node_data.get("road_node_id"),
                "original_node_id": node_data.get("original_node_id"),
                "x": node_data.get("x"),
                "y": node_data.get("y"),
                "unique_neighbor_count": len(unique_neighbors),
                "neighbor_node": str(next(iter(unique_neighbors))),
                "in_degree": G.in_degree(node),
                "out_degree": G.out_degree(node),
                "connected_edge_count": len(incident_edges),
                "connected_road_edge_ids": stringify_unique(edge_ids),
                "connected_highway": stringify_unique(highways),
                "connected_name": stringify_unique(names),
                "connected_total_length_m": sum(lengths),
                "connected_oneway": stringify_unique(oneways),
                "connected_bridge": stringify_unique(bridges),
                "connected_tunnel": stringify_unique(tunnels),
            }
        )

    dead_end_df = pd.DataFrame(records)

    output_path = output_dir / "dead_end_nodes.csv"
    dead_end_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Dead-end nodes: {len(dead_end_df)}")
    print(f"Saved to: {output_path}")

    return dead_end_df


if __name__ == "__main__":
    export_dead_end_nodes()
