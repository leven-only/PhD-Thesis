"""固定路网环境，负责道路状态更新和路径查询。"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

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
    """无向道路。"""

    edge_id: str
    node_u: str
    node_v: str
    length_km: float
    speed_kph: float

    @property
    def current_speed_kph(self) -> float:
        return max(self.speed_kph, 1e-6)

    @property
    def travel_time_seconds(self) -> float:
        return self.length_km / self.current_speed_kph * 3600


class RoadNetwork:
    """拓扑固定；道路属性只能在 step 中随环境事件更新。"""

    def __init__(
        self,
        nodes: Optional[list[RoadNode]] = None,
        edges: Optional[list[RoadEdge]] = None,
    ) -> None:
        self._nodes: dict[str, RoadNode] = {}
        self._edges: dict[str, RoadEdge] = {}
        self._graph = nx.Graph()

        for node in nodes or []:
            if node.node_id in self._nodes:
                raise ValueError(f"节点 ID 已存在: {node.node_id}")

            self._nodes[node.node_id] = deepcopy(node)
            self._graph.add_node(node.node_id)

        for edge in edges or []:
            if edge.edge_id in self._edges:
                raise ValueError(f"边 ID 已存在: {edge.edge_id}")
            if edge.node_u not in self._nodes or edge.node_v not in self._nodes:
                raise ValueError(f"道路端点不存在: {edge.node_u} - {edge.node_v}")

            self._edges[edge.edge_id] = deepcopy(edge)
            self._graph.add_edge(
                edge.node_u,
                edge.node_v,
                edge_id=edge.edge_id,
            )

        self._initial_edges = deepcopy(self._edges)

    def reset(self) -> None:
        """恢复道路的初始属性。"""
        self._edges = deepcopy(self._initial_edges)

    def step(
        self,
        time_step: float,
        current_time: float,
        action: Optional[Any] = None,
    ) -> None:
        """处理当前时间步的道路更新事件。"""
        if not isinstance(action, dict):
            return

        road_updates = action.get("road_updates")
        if not isinstance(road_updates, dict):
            return

        for edge_id, updates in road_updates.items():
            if edge_id not in self._edges:
                raise ValueError(f"道路不存在: {edge_id}")
            if not isinstance(updates, dict):
                raise TypeError(f"道路更新必须是字典: {edge_id}")

            unknown = set(updates) - {"length_km", "speed_kph"}
            if unknown:
                names = ", ".join(sorted(unknown))
                raise AttributeError(f"不允许更新道路属性: {names}")

            edge = self._edges[edge_id]
            for name, value in updates.items():
                parsed_value = float(value)
                if parsed_value < 0:
                    raise ValueError(
                        f"道路属性不能为负数: {name}={parsed_value}"
                    )

                setattr(edge, name, parsed_value)

    def get_state(self) -> dict[str, Any]:
        """返回完整路网状态副本。"""
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "nodes": {
                node_id: deepcopy(vars(node))
                for node_id, node in self._nodes.items()
            },
            "edges": {
                edge_id: {
                    **deepcopy(vars(edge)),
                    "current_speed_kph": edge.current_speed_kph,
                    "travel_time_seconds": edge.travel_time_seconds,
                }
                for edge_id, edge in self._edges.items()
            },
        }

    def get_edge_between(self, node_u: str, node_v: str) -> RoadEdge:
        """查询两个相邻节点之间的道路并返回副本。"""
        edge_data = self._graph.get_edge_data(node_u, node_v)
        if edge_data is None:
            raise ValueError(f"节点之间不存在边: {node_u} - {node_v}")

        return deepcopy(self._edges[edge_data["edge_id"]])

    def get_neighbors(self, node_id: str) -> list[str]:
        """查询一个节点的全部相邻节点。"""
        if node_id not in self._graph:
            return []

        return list(self._graph.neighbors(node_id))

    def path_distance_km(self, path: list[str]) -> float:
        """查询一条路径的总距离。"""
        return sum(
            self.get_edge_between(path[index], path[index + 1]).length_km
            for index in range(len(path) - 1)
        )

    def path_travel_time_seconds(self, path: list[str]) -> float:
        """查询一条路径在当前路况下的总通行时间。"""
        return sum(
            self.get_edge_between(
                path[index],
                path[index + 1],
            ).travel_time_seconds
            for index in range(len(path) - 1)
        )

    def shortest_path(
        self,
        origin_node_id: str,
        destination_node_id: str,
        weight: str = "time",
    ) -> list[str]:
        """按距离或当前通行时间查询最短路径。"""
        if origin_node_id not in self._nodes:
            raise ValueError(f"起点不存在: {origin_node_id}")
        if destination_node_id not in self._nodes:
            raise ValueError(f"终点不存在: {destination_node_id}")
        if weight not in ("time", "distance"):
            raise ValueError(f"不支持的路径权重: {weight}")

        def edge_weight(node_u: str, node_v: str, data: dict[str, Any]) -> float:
            edge = self._edges[data["edge_id"]]
            return (
                edge.length_km
                if weight == "distance"
                else edge.travel_time_seconds
            )

        try:
            return nx.shortest_path(
                self._graph,
                source=origin_node_id,
                target=destination_node_id,
                weight=edge_weight,
            )
        except nx.NetworkXNoPath as exc:
            raise ValueError(
                f"无法从 {origin_node_id} 到达 {destination_node_id}"
            ) from exc
