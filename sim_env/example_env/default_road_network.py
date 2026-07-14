"""默认路网样例。"""

from sim_env.core_road_network import RoadEdge, RoadNetwork, RoadNode


def build_default_road_network() -> RoadNetwork:
    """创建一个 10 节点的默认测试路网。"""
    nodes = [
        RoadNode("node_01"),
        RoadNode("node_02"),
        RoadNode("node_03"),
        RoadNode("node_04"),
        RoadNode("node_05"),
        RoadNode("node_06"),
        RoadNode("node_07"),
        RoadNode("node_08"),
        RoadNode("node_09"),
        RoadNode("node_10"),
    ]

    edges = [
        RoadEdge("edge_01", "node_01", "node_02", 1.2, 45.0),
        RoadEdge("edge_02", "node_02", "node_03", 1.1, 45.0),
        RoadEdge("edge_03", "node_03", "node_04", 1.3, 50.0),
        RoadEdge("edge_04", "node_01", "node_05", 1.0, 35.0),
        RoadEdge("edge_05", "node_02", "node_06", 1.0, 35.0),
        RoadEdge("edge_06", "node_03", "node_07", 1.0, 35.0),
        RoadEdge("edge_07", "node_04", "node_08", 1.0, 35.0),
        RoadEdge("edge_08", "node_05", "node_06", 1.2, 40.0),
        RoadEdge("edge_09", "node_06", "node_07", 1.1, 40.0),
        RoadEdge("edge_10", "node_07", "node_08", 1.2, 40.0),
        RoadEdge("edge_11", "node_05", "node_09", 1.4, 30.0),
        RoadEdge("edge_12", "node_06", "node_09", 1.0, 30.0),
        RoadEdge("edge_13", "node_07", "node_10", 1.0, 30.0),
        RoadEdge("edge_14", "node_08", "node_10", 1.4, 30.0),
        RoadEdge("edge_15", "node_09", "node_10", 1.1, 35.0),
        RoadEdge("edge_16", "node_06", "node_10", 1.6, 45.0),
    ]

    return RoadNetwork(nodes=nodes, edges=edges)
