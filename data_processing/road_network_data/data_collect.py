from shapely.geometry import Polygon
import osmnx as ox
import networkx as nx
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RAW_NETWORK_DIR = BASE_DIR / "raw_network"
CACHE_DIR = BASE_DIR / "cache"

EDGES_CSV_PATH = RAW_NETWORK_DIR / "beijing_sixth_ring_core_edges.csv"
EDGES_GEOJSON_PATH = RAW_NETWORK_DIR / "beijing_sixth_ring_core_edges.geojson"
NODES_GEOJSON_PATH = RAW_NETWORK_DIR / "beijing_sixth_ring_core_nodes.geojson"
GRAPHML_PATH = RAW_NETWORK_DIR / "beijing_sixth_ring_core_network.graphml"

study_area_polygon = Polygon([
    (116.192888, 39.760024),  # 左下角 west, south
    (116.563710, 39.760024),  # 右下角 east, south
    (116.563710, 40.035952),  # 右上角 east, north
    (116.192888, 40.035952),  # 左上角 west, north
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
    ox.settings.cache_folder = CACHE_DIR
    RAW_NETWORK_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("开始下载 OSM 路网数据...")

    # 1. 先下载六环范围内全部机动车路网
    G = ox.graph_from_polygon(
        study_area_polygon,
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
    print("筛选后核心路段数量:", len(edges_to_keep))

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
        EDGES_CSV_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    edges.to_file(
        EDGES_GEOJSON_PATH,
        driver="GeoJSON"
    )

    nodes.to_file(
        NODES_GEOJSON_PATH,
        driver="GeoJSON"
    )

    ox.save_graphml(
        G_core,
        GRAPHML_PATH
    )

    print("数据保存完成:")
    print(f"- {EDGES_CSV_PATH}")
    print(f"- {EDGES_GEOJSON_PATH}")
    print(f"- {NODES_GEOJSON_PATH}")
    print(f"- {GRAPHML_PATH}")

    return G_core, nodes, edges


if __name__ == "__main__":
    download_road_network()
