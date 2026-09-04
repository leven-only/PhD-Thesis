from bisect import bisect_right
from typing import Any

import networkx as nx


class RoadNetwork:
    def __init__(
        self,
        matrix,
        speed_matrix: Any,
        speed_timetable: Any,
        start_time: float,
    ) -> None:
        self.matrix = matrix    # 路网矩阵，0表示不连接，1表示连接
        self.speed_matrix = speed_matrix    # 速度矩阵：每行代表一个速度的snapshot
        self.speed_timetable = speed_timetable  # 时间戳数组，元素数量与speed_matrix 行数相同，每个元素代表当时的时间
        self.start_time = start_time    # 场景开始时间

        self.node_count = len(self.matrix)  # 节点数量
        self.graph = None  # 有向图对象：路网的核心数据结构，节点为路网节点，边为道路及其属性（edge_id/length_km/speed_kph/travel_time_seconds），由 _initialize_network() 构建
        self._edge_positions = []  # 边位置列表：邻接矩阵中所有非零元素的(行, 列)坐标，即全部道路的(起点, 终点)编号，顺序决定 speed_matrix 每一列对应哪条道路，由 _initialize_network() 填充
        self.current_speed_snapshot_index = None  # 当前速度快照下标：speed_matrix/speed_timetable 中正在生效的行下标，由 _apply_speed_snapshot() 更新（reset()/step() 内部调用）

        self._initialize_network()

    # ------------------------------------------------------------------
    # 标准状态接口：reset / step / get_state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """把所有道路速度恢复到 start_time 对应的速度快照。"""
        start_time = self._system_time2speed_index(self.start_time)
        self._apply_speed_snapshot(start_time)

    def step(self, current_time: float) -> None:
        """根据当前仿真时间刷新道路速度；如果快照没变则什么都不做。"""
        snapshot_index = self._system_time2speed_index(current_time)
        if snapshot_index != self.current_speed_snapshot_index:
            self._apply_speed_snapshot(snapshot_index)

    def get_state(self) -> dict[str, Any]:
        """返回当前路网状态：节点、道路数量和每条道路的当前属性。"""
        return {
            "nodes": list(self.graph.nodes),
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "current_speed_snapshot_index": self.current_speed_snapshot_index,
            "current_speed_time": self.speed_timetable[self.current_speed_snapshot_index],
            "edges": [
                {"node_u": node_u, "node_v": node_v, **dict(data)}
                for node_u, node_v, data in self.graph.edges(data=True)
            ],
        }

    # ------------------------------------------------------------------
    # 个性化查询接口
    # ------------------------------------------------------------------

    def get_edge_between(self, node_u: int, node_v: int) -> dict[str, Any]:
        """
        返回节点之间路段信息
        """
        try:
            return self.graph.edges[node_u, node_v]
        except (KeyError, nx.NetworkXError) as exc:
            raise ValueError(f"节点之间不存在道路: {node_u} -> {node_v}") from exc

    # ------------------------------------------------------------------
    # 内部函数，不向外部提供
    # ------------------------------------------------------------------

    def _initialize_network(self) -> None:
        """根据邻接矩阵创建有向图，并记录非零元素位置以便后续更新速度。"""
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(self.node_count))

        self._edge_positions = [
            (row, column)
            for row in range(self.node_count)
            for column in range(self.node_count)
            if self.matrix[row][column] != 0
        ]
        for index, (row, column) in enumerate(self._edge_positions):
            self.graph.add_edge(
                row,
                column,
                edge_id=f"edge_{index:04d}",
                length_km=self.matrix[row][column],
                speed_kph=0.0,
                travel_time_seconds=0.0,
            )

    def _apply_speed_snapshot(self, snapshot_index: int) -> None:
        """
        更新边属性   距离、速度（km/h）、通行时间（s）
        更新self属性    当前速度快照下标
        """
        speeds = self.speed_matrix[snapshot_index]
        for (row, column), speed in zip(self._edge_positions, speeds):
            edge_data = self.graph.edges[row, column]
            edge_data["speed_kph"] = speed
            edge_data["travel_time_seconds"] = edge_data["length_km"] / max(speed, 1e-6) * 3600
        self.current_speed_snapshot_index = snapshot_index

    def _system_time2speed_index(self, current_time: float) -> int:
        """
        根据当前系统时间，计算在 speed_timetable 中找到最接近当前系统时间的左侧时间
        index 用于获取当前时间的速度快中
        """""
        return max(bisect_right(self.speed_timetable, current_time) - 1, 0)