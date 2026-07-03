"""路网模块，用于加载道路节点与边，并提供路径查询、行驶时间和交通状态相关能力。"""

from dataclasses import dataclass
from typing import Any, Optional, cast

import networkx as nx


@dataclass
class RoadNode:
    """路网节点。"""

    node_id: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    name: Optional[str] = None


@dataclass
class RoadEdge:
    """无向路网边。"""

    edge_id: str
    node_u: str
    node_v: str
    length_km: float
    speed_kph: float

    @property
    def current_speed_kph(self) -> float:
        """返回当前速度。"""
        return max(self.speed_kph, 1e-6)

    @property
    def travel_time_seconds(self) -> float:
        """返回当前交通状态下通过该边所需时间。"""
        return self.length_km / self.current_speed_kph * 3600


class RoadNetwork:
    """简单无向路网环境。"""

    def __init__(
        self,
        nodes: Optional[list[RoadNode]] = None,
        edges: Optional[list[RoadEdge]] = None,
    ) -> None:
        self.nodes: dict[str, RoadNode] = {}
        self.edges: dict[str, RoadEdge] = {}
        self.graph = nx.Graph()

        if nodes is None and edges is None:
            nodes, edges = self._build_default_network()

        for node in nodes or []:
            self.add_node(node)

        for edge in edges or []:
            self.add_edge(edge)

    # 同步更新graph信息
    def _sync_graph_edge(self, edge_id: str) -> None:
        """将 RoadEdge 的动态属性同步到 networkx.Graph。"""
        edge = self.edges[edge_id]
        graph_edge = cast(dict[str, Any], self.graph[edge.node_u][edge.node_v])
        graph_edge["speed_kph"] = edge.speed_kph
        graph_edge["current_speed_kph"] = edge.current_speed_kph
        graph_edge["travel_time_seconds"] = edge.travel_time_seconds

    # 添加路网节点
    def add_node(self, node: RoadNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"节点 ID 已存在: {node.node_id}")

        self.nodes[node.node_id] = node
        self.graph.add_node(
            node.node_id,
            name=node.name,
            node=node,
        )

    # 添加无向边
    def add_edge(self, edge: RoadEdge) -> None:
        if edge.edge_id in self.edges:
            raise ValueError(f"边 ID 已存在: {edge.edge_id}")

        if edge.node_u not in self.nodes:
            raise ValueError(f"边的起点不存在: {edge.node_u}")

        if edge.node_v not in self.nodes:
            raise ValueError(f"边的终点不存在: {edge.node_v}")

        self.edges[edge.edge_id] = edge
        self.graph.add_edge(
            edge.node_u,
            edge.node_v,
            edge_id=edge.edge_id,
            length_km=edge.length_km,
            speed_kph=edge.speed_kph,
            current_speed_kph=edge.current_speed_kph,
            travel_time_seconds=edge.travel_time_seconds,
            edge=edge,
        )

    # 重置路网状态。
    def reset(self) -> None:
        """
        当前版本不在路网内部保存初始速度备份。
        后续速度状态由外部交通数据导入并通过 update_speeds 更新。
        """

    # 推进一个时间步
    def step(self, time_step: float, current_time: float, action: Optional[Any] = None) -> None:
        """推进路网一个时间步。

        初始版本只根据 action 更新边速度，不主动生成交通波动。
        """
        if isinstance(action, dict) and "edge_speeds_kph" in action:
            self.update_speeds(action["edge_speeds_kph"])

    # 批量更新边的当前速度。
    def update_speeds(self, edge_speeds_kph: dict[str, float]) -> None:
        for edge_id, speed_kph in edge_speeds_kph.items():
            if edge_id not in self.edges:
                continue

            self.edges[edge_id].speed_kph = max(float(speed_kph), 0.0)
            self._sync_graph_edge(edge_id)

    # 返回两个相邻节点之间的边。
    def get_edge_between(self, node_u: str, node_v: str) -> RoadEdge:
        edge_data = self.graph.get_edge_data(node_u, node_v)

        if edge_data is None:
            raise ValueError(f"节点之间不存在边: {node_u} - {node_v}")

        edge_id = edge_data["edge_id"]
        return self.edges[edge_id]

    # 返回节点的相邻节点。
    def get_neighbors(self, node_id: str) -> list[str]:
        if node_id not in self.graph:
            return []

        return list(self.graph.neighbors(node_id))

    # 返回当前路网图的副本，避免外部直接修改内部图。
    def get_graph(self) -> nx.Graph:
        return self.graph.copy()

    # 获取路段距离
    def path_distance_km(self, path: list[str]) -> float:
        """计算路径总距离。"""
        return sum(
            self.get_edge_between(path[i], path[i + 1]).length_km
            for i in range(len(path) - 1)
        )

    # 计算路径当前通行时间。
    def path_travel_time_seconds(self, path: list[str]) -> float:
        return sum(
            self.get_edge_between(path[i], path[i + 1]).travel_time_seconds
            for i in range(len(path) - 1)
        )

    # 返回路网状态。
    def get_state(self) -> dict[str, Any]:
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": {
                node_id: {
                    "name": node.name,
                }
                for node_id, node in self.nodes.items()
            },
            "edges": {
                edge_id: {
                    "node_u": edge.node_u,
                    "node_v": edge.node_v,
                    "length_km": edge.length_km,
                    "speed_kph": edge.speed_kph,
                    "current_speed_kph": edge.current_speed_kph,
                    "travel_time_seconds": edge.travel_time_seconds,
                }
                for edge_id, edge in self.edges.items()
            },
        }

    # 计算最短路径。
    def shortest_path(self, origin_node_id: str, destination_node_id: str, weight: str = "time") -> list[str]:
        if origin_node_id not in self.nodes:
            raise ValueError(f"起点不存在: {origin_node_id}")

        if destination_node_id not in self.nodes:
            raise ValueError(f"终点不存在: {destination_node_id}")

        if weight not in ("time", "distance"):
            raise ValueError(f"不支持的路径权重: {weight}")

        graph_weight = "length_km" if weight == "distance" else "travel_time_seconds"

        try:
            return nx.shortest_path(
                self.graph,
                source=origin_node_id,
                target=destination_node_id,
                weight=graph_weight,
            )
        except nx.NetworkXNoPath as exc:
            raise ValueError(f"无法从 {origin_node_id} 到达 {destination_node_id}")

    # 构建默认路网
    def _build_default_network(self) -> tuple[list[RoadNode], list[RoadEdge]]:
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

        return nodes, edges
