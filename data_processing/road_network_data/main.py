import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

sixth_ring_polygon = Polygon([
    (116.05, 39.68),
    (116.75, 39.68),
    (116.75, 40.18),
    (116.05, 40.18),
])

target_road_types = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link"
}

def is_target_road(data):
    highway = data.get("highway")

    # highway 有时是 list，有时是 str
    if isinstance(highway, list):
        return any(h in target_road_types for h in highway)
    else:
        return highway in target_road_types


def download_road_network():
    ox.settings.use_cache = True
    ox.settings.log_console = True

    # 1. 先下载六环范围内全部机动车路网
    G = ox.graph_from_polygon(
        sixth_ring_polygon,
        network_type="drive",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True
    )

    # 2. 只保留环路/快速路等级道路
    edges_to_keep = [
        (u, v, k)
        for u, v, k, data in G.edges(keys=True, data=True)
        if is_target_road(data)
    ]

    G_core = G.edge_subgraph(edges_to_keep).copy()

    # 3. 删除孤立节点
    G_core.remove_nodes_from(list(nx.isolates(G_core)))

    # 4. 添加速度和行驶时间
    G_core = ox.add_edge_speeds(G_core)
    G_core = ox.add_edge_travel_times(G_core)

    # 5. 转为 GeoDataFrame
    nodes, edges = ox.graph_to_gdfs(G_core)

    print("核心节点数量:", len(nodes))
    print("核心路段数量:", len(edges))
    print(edges["highway"].value_counts())

    # 6. 保存数据
    edge_data = edges.reset_index()[[
        "u",
        "v",
        "key",
        "osmid",
        "highway",
        "oneway",
        "length",
        "speed_kph",
        "travel_time",
        "geometry"
    ]]

    edge_data.to_csv(
        "beijing_sixth_ring_core_edges.csv",
        index=False,
        encoding="utf-8-sig"
    )

    edges.to_file(
        "beijing_sixth_ring_core_edges.geojson",
        driver="GeoJSON"
    )

    nodes.to_file(
        "beijing_sixth_ring_core_nodes.geojson",
        driver="GeoJSON"
    )

    ox.save_graphml(
        G_core,
        "beijing_sixth_ring_core_network.graphml"
    )

    return G_core, nodes, edges

def draw_road_network(
    graphml_path="beijing_sixth_ring_core_network.graphml",
    output_path="beijing_sixth_ring_core_network.jpg",
    dpi=600
):
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
    print("road network data collection and drawing")
    # download_road_network()
    draw_road_network()