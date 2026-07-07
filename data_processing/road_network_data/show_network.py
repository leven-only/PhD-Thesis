import osmnx as ox
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_NETWORK_DIR = BASE_DIR / "raw_network"
RAW_GRAPHML_PATH = RAW_NETWORK_DIR / "beijing_sixth_ring_core_network.graphml"
RAW_NETWORK_FIGURE_PATH = RAW_NETWORK_DIR / "beijing_sixth_ring_core_network.jpg"

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

if __name__ == '__main__':
    draw_road_network()
