"""
给节点和边建立可追踪标识
"""
from pathlib import Path

import osmnx as ox
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

RAW_GRAPHML_PATH = (
    BASE_DIR
    / "raw_network"
    / "beijing_sixth_ring_core_network.graphml"
)

OUTPUT_DIR = (
    BASE_DIR
    / "processed_network"
    / "step_02_with_ids"
)


def add_traceable_ids(
    graphml_path=RAW_GRAPHML_PATH,
    output_dir=OUTPUT_DIR,
):
    """
    Add traceable internal IDs to all road nodes and road edges.

    This step does not modify the raw network files.
    It creates a new graph and exported tables with additional ID fields.

    Node ID rule:
        road_node_id = node_000001, node_000002, ...

    Edge ID rule:
        road_edge_id = edge_000001, edge_000002, ...

    Original IDs are preserved for traceability.
    """

    graphml_path = Path(graphml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not graphml_path.exists():
        raise FileNotFoundError(f"GraphML file not found: {graphml_path}")

    G = ox.load_graphml(graphml_path)

    original_node_count = G.number_of_nodes()
    original_edge_count = G.number_of_edges()

    original_total_length = sum(
        float(data.get("length", 0))
        for _, _, _, data in G.edges(keys=True, data=True)
    )

    # ------------------------------------------------------------
    # 1. Add node IDs
    # ------------------------------------------------------------
    node_id_map = {}

    for index, node in enumerate(G.nodes, start=1):
        road_node_id = f"node_{index:06d}"
        node_id_map[node] = road_node_id

        G.nodes[node]["road_node_id"] = road_node_id
        G.nodes[node]["original_node_id"] = str(node)

    # ------------------------------------------------------------
    # 2. Add edge IDs
    # ------------------------------------------------------------
    edge_mapping_records = []

    for index, (u, v, key, data) in enumerate(
        G.edges(keys=True, data=True),
        start=1,
    ):
        road_edge_id = f"edge_{index:06d}"

        data["road_edge_id"] = road_edge_id
        data["original_u"] = str(u)
        data["original_v"] = str(v)
        data["original_key"] = str(key)
        data["original_osmid"] = str(data.get("osmid", ""))
        data["road_node_u"] = node_id_map[u]
        data["road_node_v"] = node_id_map[v]

        edge_mapping_records.append(
            {
                "road_edge_id": road_edge_id,
                "original_u": str(u),
                "original_v": str(v),
                "original_key": str(key),
                "original_osmid": str(data.get("osmid", "")),
                "road_node_u": node_id_map[u],
                "road_node_v": node_id_map[v],
            }
        )

    # ------------------------------------------------------------
    # 3. Convert to GeoDataFrames
    # ------------------------------------------------------------
    nodes, edges = ox.graph_to_gdfs(G)

    # ------------------------------------------------------------
    # 4. Create mapping tables
    # ------------------------------------------------------------
    node_mapping = pd.DataFrame(
        [
            {
                "road_node_id": road_node_id,
                "original_node_id": str(original_node_id),
                "x": G.nodes[original_node_id].get("x"),
                "y": G.nodes[original_node_id].get("y"),
            }
            for original_node_id, road_node_id in node_id_map.items()
        ]
    )

    edge_mapping = pd.DataFrame(edge_mapping_records)

    # ------------------------------------------------------------
    # 5. Export graph, GeoJSON, CSV, and mapping tables
    # ------------------------------------------------------------
    ox.save_graphml(
        G,
        filepath=output_dir / "network_with_ids.graphml",
    )

    nodes.to_file(
        output_dir / "nodes_with_ids.geojson",
        driver="GeoJSON",
    )

    edges.to_file(
        output_dir / "edges_with_ids.geojson",
        driver="GeoJSON",
    )

    nodes.reset_index().to_csv(
        output_dir / "nodes_with_ids.csv",
        index=False,
        encoding="utf-8-sig",
    )

    edges.reset_index().to_csv(
        output_dir / "edges_with_ids.csv",
        index=False,
        encoding="utf-8-sig",
    )

    node_mapping.to_csv(
        output_dir / "node_id_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )

    edge_mapping.to_csv(
        output_dir / "edge_id_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------
    # 6. Validation summary
    # ------------------------------------------------------------
    new_node_count = G.number_of_nodes()
    new_edge_count = G.number_of_edges()

    new_total_length = sum(
        float(data.get("length", 0))
        for _, _, _, data in G.edges(keys=True, data=True)
    )

    print("=" * 60)
    print("STEP 02: ADD TRACEABLE IDS")
    print("=" * 60)
    print(f"Input graph: {graphml_path}")
    print(f"Output directory: {output_dir}")
    print(f"Nodes: {original_node_count} -> {new_node_count}")
    print(f"Edges: {original_edge_count} -> {new_edge_count}")
    print(f"Total length: {original_total_length:.2f} m -> {new_total_length:.2f} m")
    print("Finished.")

    return G, nodes, edges, node_mapping, edge_mapping


if __name__ == "__main__":
    pass