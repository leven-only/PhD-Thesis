import osmnx as ox
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_NETWORK_DIR = BASE_DIR / "raw_network"
RAW_GRAPHML_PATH = RAW_NETWORK_DIR / "beijing_sixth_ring_core_network.graphml"
RAW_NETWORK_FIGURE_PATH = RAW_NETWORK_DIR / "beijing_sixth_ring_core_network.jpg"
CANDIDATE_NODES_PATH = BASE_DIR / "candidate_node"
LOCAL_NETWORK_OUTPUT_DIR = BASE_DIR / "node_local_network_checks"
LOCAL_NETWORK_FIGURE_PATH = LOCAL_NETWORK_OUTPUT_DIR / "selected_nodes_3km_local_network.jpg"
LOCAL_NETWORK_SUMMARY_PATH = LOCAL_NETWORK_OUTPUT_DIR / "selected_nodes_snap_summary.csv"
LOCAL_NETWORK_MARKER_PATH = LOCAL_NETWORK_OUTPUT_DIR / "selected_nodes_trace_markers.csv"

"""
Raw Network
"""
def inspect_raw_network(
    graphml_path=RAW_GRAPHML_PATH
):
    """
    Inspect the basic structure and available attributes of the raw road network.

    This function does not modify the graph. It only reports:
    - graph attributes
    - number of nodes and edges
    - available node and edge fields
    - road type distribution
    - total road length
    """

    graphml_path = Path(graphml_path)

    if not graphml_path.exists():
        raise FileNotFoundError(f"GraphML file not found: {graphml_path}")

    # Load graph
    G = ox.load_graphml(graphml_path)

    # Convert to GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G)

    print("=" * 60)
    print("RAW ROAD NETWORK BASIC INFORMATION")
    print("=" * 60)

    # 1. Graph attributes
    print("\n[1] Graph attributes")
    print("-" * 60)
    for key, value in G.graph.items():
        print(f"{key}: {value}")

    # 2. Basic graph size
    print("\n[2] Graph size")
    print("-" * 60)
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Is directed: {G.is_directed()}")
    print(f"Is multigraph: {G.is_multigraph()}")

    # 3. Node attributes
    print("\n[3] Node attributes")
    print("-" * 60)
    print(f"Node table shape: {nodes.shape}")
    print("Available node fields:")
    for col in nodes.columns:
        print(f"- {col}")

    # 4. Edge attributes
    print("\n[4] Edge attributes")
    print("-" * 60)
    print(f"Edge table shape: {edges.shape}")
    print("Available edge fields:")
    for col in edges.columns:
        print(f"- {col}")

    # 5. Road type distribution
    print("\n[5] Road type distribution")
    print("-" * 60)

    if "highway" in edges.columns:
        highway_series = edges["highway"].astype(str)

        road_type_summary = (
            highway_series
            .value_counts()
            .reset_index()
        )
        road_type_summary.columns = ["highway", "edge_count"]

        if "length" in edges.columns:
            length_summary = (
                edges.assign(highway=edges["highway"].astype(str))
                .groupby("highway")["length"]
                .sum()
                .reset_index()
                .rename(columns={"length": "total_length_m"})
            )

            road_type_summary = road_type_summary.merge(
                length_summary,
                on="highway",
                how="left"
            )

            road_type_summary["length_percentage"] = (
                road_type_summary["total_length_m"]
                / road_type_summary["total_length_m"].sum()
                * 100
            )

        road_type_summary["edge_percentage"] = (
            road_type_summary["edge_count"]
            / road_type_summary["edge_count"].sum()
            * 100
        )

        print(road_type_summary.to_string(index=False))
    else:
        print("No 'highway' field found in edge attributes.")

    # 6. Length information
    print("\n[6] Road length statistics")
    print("-" * 60)

    if "length" in edges.columns:
        print(f"Total road length: {edges['length'].sum():.2f} m")
        print(f"Average edge length: {edges['length'].mean():.2f} m")
        print(f"Minimum edge length: {edges['length'].min():.2f} m")
        print(f"Maximum edge length: {edges['length'].max():.2f} m")
    else:
        print("No 'length' field found in edge attributes.")

    print("\nInspection finished.")

    return G, nodes, edges




"""
=========
路网图像绘制
=========
"""
def draw_road_network(
    graphml_path=RAW_GRAPHML_PATH,
    output_path=RAW_NETWORK_FIGURE_PATH,
    dpi=600
):
    graphml_path = Path(graphml_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not graphml_path.exists():
        raise FileNotFoundError(f"GraphML file not found: {graphml_path}")

    # 1. 读取路网
    G = ox.load_graphml(graphml_path)

    # 2. 绘制路网
    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_linewidth=1.0,
        edge_color="black",
        bgcolor="white",
        show=False,
        close=False
    )

    # 3. 去除边框和坐标轴
    ax.set_axis_off()

    # 4. 保存为高清 JPG
    plt.savefig(
        output_path,
        dpi=dpi,
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.02
    )

    plt.close(fig)

    print(f"Road network figure saved to: {output_path}")


"""
===========================
节点周边可达路网检查图像绘制
===========================
"""
def haversine_distance_m(lat1, lon1, lat2, lon2):
    earth_radius_m = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * earth_radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_candidate_nodes(node_list_path=CANDIDATE_NODES_PATH):
    """
    Load candidate nodes from a same-level candidate_node file.

    Default format:
        lon,lat
        116.349254,39.784470

    A headered CSV with node_id, lat, lon is also supported for later reuse.
    """

    node_list_path = Path(node_list_path)

    if not node_list_path.exists():
        raise FileNotFoundError(f"Candidate node file not found: {node_list_path}")

    first_line = node_list_path.read_text(encoding="utf-8-sig").splitlines()[0]
    first_values = [value.strip().lower() for value in first_line.split(",")]

    if {"node_id", "lat", "lon"}.issubset(first_values):
        selected_nodes = pd.read_csv(node_list_path)
        return selected_nodes[["node_id", "lat", "lon"]]

    selected_nodes = pd.read_csv(
        node_list_path,
        header=None,
        names=["lon", "lat"],
    )
    selected_nodes.insert(0, "node_id", range(1, len(selected_nodes) + 1))

    return selected_nodes[["node_id", "lat", "lon"]]


def draw_selected_nodes_local_network(
    node_list_path=CANDIDATE_NODES_PATH,
    graphml_path=RAW_GRAPHML_PATH,
    output_path=LOCAL_NETWORK_FIGURE_PATH,
    summary_path=LOCAL_NETWORK_SUMMARY_PATH,
    marker_path=LOCAL_NETWORK_MARKER_PATH,
    radius_m=3000,
    dpi=600
):
    """
    Draw one combined figure for selected simulation nodes and their local OSM road
    network neighborhoods.

    By default, this function reads the same-level candidate_node file, whose
    format is one lon,lat pair per line without a header. Node IDs are generated
    by row order. A headered CSV with node_id,lat,lon is also supported.

    For each selected node, this function:
    - matches the selected coordinate to its nearest OSM edge;
    - uses the two endpoints of that edge as topological tracing starts;
    - extracts the OSM road network reachable within radius_m along road length;
    - merges all local ego networks into one combined subgraph;
    - marks selected coordinates in red, start endpoints in blue, dead ends in
      orange, and cutoff frontier nodes in purple.
    """

    node_list_path = Path(node_list_path)
    graphml_path = Path(graphml_path)
    output_path = Path(output_path)
    summary_path = Path(summary_path)
    marker_path = Path(marker_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.parent.mkdir(parents=True, exist_ok=True)

    if not graphml_path.exists():
        raise FileNotFoundError(f"GraphML file not found: {graphml_path}")

    selected_nodes = load_candidate_nodes(node_list_path)

    G = ox.load_graphml(graphml_path)
    G_trace = ox.convert.to_undirected(G)

    local_node_ids = set()
    summary_records = []
    marker_records = []

    for _, row in selected_nodes.iterrows():
        simulation_node_id = row["node_id"]
        lat = float(row["lat"])
        lon = float(row["lon"])

        nearest_edge, nearest_edge_distance = ox.distance.nearest_edges(
            G,
            X=lon,
            Y=lat,
            return_dist=True,
        )

        edge_u, edge_v, edge_key = nearest_edge
        start_nodes = [edge_u, edge_v]

        trace_lengths = nx.multi_source_dijkstra_path_length(
            G_trace,
            sources=start_nodes,
            cutoff=radius_m,
            weight="length",
        )
        trace_node_ids = set(trace_lengths)
        local_node_ids.update(trace_node_ids)

        dead_end_nodes = []
        frontier_nodes = []

        for osm_node, distance_m in trace_lengths.items():
            osm_node_data = G_trace.nodes[osm_node]

            if G_trace.degree(osm_node) <= 1 and distance_m < radius_m * 0.98:
                marker_type = "dead_end"
                dead_end_nodes.append(osm_node)
            elif distance_m >= radius_m * 0.95:
                marker_type = "frontier"
                frontier_nodes.append(osm_node)
            else:
                continue

            marker_records.append({
                "node_id": simulation_node_id,
                "marker_type": marker_type,
                "osm_node": osm_node,
                "osm_lat": osm_node_data["y"],
                "osm_lon": osm_node_data["x"],
                "distance_from_start_m": distance_m,
                "osm_degree": G_trace.degree(osm_node),
            })

        edge_u_data = G.nodes[edge_u]
        edge_v_data = G.nodes[edge_v]
        distance_to_u_m = haversine_distance_m(
            lat,
            lon,
            edge_u_data["y"],
            edge_u_data["x"],
        )
        distance_to_v_m = haversine_distance_m(
            lat,
            lon,
            edge_v_data["y"],
            edge_v_data["x"],
        )

        summary_records.append({
            "node_id": simulation_node_id,
            "lat": lat,
            "lon": lon,
            "nearest_edge_u": edge_u,
            "nearest_edge_v": edge_v,
            "nearest_edge_key": edge_key,
            "nearest_edge_distance_graph_units": nearest_edge_distance,
            "edge_u_lat": edge_u_data["y"],
            "edge_u_lon": edge_u_data["x"],
            "edge_v_lat": edge_v_data["y"],
            "edge_v_lon": edge_v_data["x"],
            "distance_to_edge_u_m": distance_to_u_m,
            "distance_to_edge_v_m": distance_to_v_m,
            "trace_node_count": len(trace_node_ids),
            "dead_end_count": len(dead_end_nodes),
            "frontier_count": len(frontier_nodes),
        })

    combined_graph = G_trace.subgraph(local_node_ids).copy()

    fig, ax = ox.plot_graph(
        combined_graph,
        node_size=0,
        edge_linewidth=1.0,
        edge_color="#555555",
        bgcolor="white",
        show=False,
        close=False,
    )

    for record in summary_records:
        ax.scatter(
            record["lon"],
            record["lat"],
            s=46,
            c="red",
            marker="o",
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
        )
        for endpoint_lon, endpoint_lat in [
            (record["edge_u_lon"], record["edge_u_lat"]),
            (record["edge_v_lon"], record["edge_v_lat"]),
        ]:
            ax.scatter(
                endpoint_lon,
                endpoint_lat,
                s=42,
                c="blue",
                marker="x",
                linewidths=1.2,
                zorder=6,
            )
        ax.text(
            record["lon"],
            record["lat"],
            str(record["node_id"]),
            fontsize=8,
            color="red",
            ha="left",
            va="bottom",
            zorder=7,
        )

    for marker in marker_records:
        if marker["marker_type"] == "dead_end":
            color = "orange"
            marker_shape = "s"
            size = 18
        else:
            color = "purple"
            marker_shape = "."
            size = 10

        ax.scatter(
            marker["osm_lon"],
            marker["osm_lat"],
            s=size,
            c=color,
            marker=marker_shape,
            linewidths=0,
            zorder=4,
        )

    ax.set_axis_off()

    plt.savefig(
        output_path,
        dpi=dpi,
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)

    summary = pd.DataFrame(summary_records)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    markers = pd.DataFrame(marker_records)
    markers.to_csv(marker_path, index=False, encoding="utf-8-sig")

    print(f"Selected node local network figure saved to: {output_path}")
    print(f"Selected node snap summary saved to: {summary_path}")
    print(f"Selected node trace markers saved to: {marker_path}")

    return combined_graph, summary, markers


if __name__ == '__main__':
    draw_selected_nodes_local_network()
