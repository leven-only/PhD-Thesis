"""路网组件：从邻接矩阵和速度快照建立路网。"""

from bisect import bisect_right
from typing import Any

import networkx as nx

class RoadNetwork:
    """由邻接矩阵和速度数据构成的固定路网。

    ``matrix`` 的非零值表示道路长度（km）。矩阵按从上到下、
    从左到右扫描；每个非零元素依次对应 ``speed_matrix`` 的一列，
    ``speed_matrix`` 的每一行表示一个速度快照。
    """
    def __init__(
        self,
        matrix: Any,
        speed_matrix: Any,
        speed_timetable: Any,
        start_time: float,
    ) -> None:
        self.matrix = self._to_2_list(matrix)
        self.speed_matrix = self._to_2_list(speed_matrix)
        self.speed_timetable = self._to_1_list(speed_timetable)
        self.start_time = float(start_time)
        self._validate_data()
        self._initialize_network()
        self.reset()

    def _initialize_network(self) -> None:
        """根据邻接矩阵创建 NetworkX 有向图。"""
        # 矩阵行列下标直接作为节点 ID
        node_count = len(self.matrix)
        self.nodes = list(range(node_count))

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes)   # 添加节点

        # 记录非零元素的位置，顺序就是速度数据的列顺序，方便速度更新
        self._edge_positions = [
            (row, column)
            for row in range(node_count)
            for column in range(node_count)
            if self.matrix[row][column] != 0
        ]

        # 将道路及其属性直接加入图中
        for index, (row, column) in enumerate(self._edge_positions):
            node_u = self.nodes[row]    # 起点
            node_v = self.nodes[column] # 终点
            self.graph.add_edge(
                node_u,
                node_v,
                edge_id=f"edge_{index:04d}",
                length_km=self.matrix[row][column],
                speed_kph=0.0,
                travel_time_seconds=0.0,
            )

    def reset(self) -> None:
        """恢复到环境开始时间对应的速度快照。"""
        self._set_speed_snapshot(self._get_snapshot_index(self.start_time))

    def step(
        self,
        current_time: float,
    ):
        """根据当前仿真时间更新道路速度。"""
        snapshot_index = self._get_snapshot_index(current_time)

        if snapshot_index != self.current_speed_snapshot_index:
            self._set_speed_snapshot(snapshot_index)

    def get_state(self) -> dict[str, Any]:
        """返回当前路网状态。"""
        return {
            "nodes": list(self.graph.nodes),
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "current_speed_snapshot_index": self.current_speed_snapshot_index,
            "current_speed_time": self.speed_timetable[
                self.current_speed_snapshot_index
            ],
            "edges": [
                {"node_u": node_u, "node_v": node_v, **dict(data)}
                for node_u, node_v, data in self.graph.edges(data=True)
            ],
        }

    def get_edge_between(self, node_u: int, node_v: int) -> dict[str, Any]:
        """返回两个节点之间的道路，供移动模块使用。"""
        try:
            return self.graph.edges[node_u, node_v]
        except (KeyError, nx.NetworkXError) as exc:
            raise ValueError(f"节点之间不存在道路: {node_u} -> {node_v}") from exc

    def _set_speed_snapshot(self, snapshot_index: int) -> None:
        """更新速度快照"""
        speeds = self.speed_matrix[snapshot_index]
        for (row, column), speed in zip(self._edge_positions, speeds):
            node_u = self.nodes[row]
            node_v = self.nodes[column]
            edge_data = self.graph.edges[node_u, node_v]
            edge_data["speed_kph"] = speed
            edge_data["travel_time_seconds"] = (
                edge_data["length_km"] / max(speed, 1e-6) * 3600
            )
        self.current_speed_snapshot_index = snapshot_index

    def _get_snapshot_index(self, current_time: float) -> int:
        """返回不晚于当前时间的最近速度快照下标。"""
        return max(bisect_right(self.speed_timetable, current_time) - 1, 0)

    def _validate_data(self) -> None:
        node_count = len(self.matrix)
        if node_count == 0 or any(len(row) != node_count for row in self.matrix):
            raise ValueError("matrix 必须是非空方阵")
        if any(value < 0 for row in self.matrix for value in row):
            raise ValueError("matrix 不能包含负数")
        if any(self.matrix[index][index] != 0 for index in range(node_count)):
            raise ValueError("matrix 对角线必须为 0")

        edge_count = sum(value != 0 for row in self.matrix for value in row)
        if edge_count == 0:
            raise ValueError("matrix 中没有道路")
        if not self.speed_matrix:
            raise ValueError("speed_matrix 不能为空")

        if len(self.speed_timetable) != len(self.speed_matrix):
            raise ValueError(
                f"speed_timetable 有 {len(self.speed_timetable)} 个时间，"
                f"但 speed_matrix 有 {len(self.speed_matrix)} 行"
            )
        if any(
            current_time >= next_time
            for current_time, next_time in zip(
                self.speed_timetable,
                self.speed_timetable[1:],
            )
        ):
            raise ValueError("speed_timetable 必须严格递增")

        for row_index, speed_row in enumerate(self.speed_matrix):
            if len(speed_row) != edge_count:
                raise ValueError(
                    f"speed_matrix 第 {row_index + 1} 行有 "
                    f"{len(speed_row)} 个速度值，但 matrix 中有 "
                    f"{edge_count} 个非零元素"
                )
        if any(value < 0 for row in self.speed_matrix for value in row):
            raise ValueError("speed_matrix 不能包含负数")

    @staticmethod
    def _to_2_list(data: Any) -> list[list[float]]:
        """把 list、NumPy 数组或 DataFrame 转成二维列表。"""
        if hasattr(data, "to_numpy"):
            data = data.to_numpy()
        if hasattr(data, "tolist"):
            data = data.tolist()

        try:
            return [[float(value) for value in row] for row in data]
        except (TypeError, ValueError) as exc:
            raise TypeError("矩阵数据必须是二维数值数据") from exc

    @staticmethod
    def _to_1_list(data: Any) -> list[float]:
        """把一列时间数据转换成一维列表。"""
        if hasattr(data, "to_numpy"):
            data = data.to_numpy()
        if hasattr(data, "tolist"):
            data = data.tolist()

        try:
            return [
                float(value[0] if isinstance(value, (list, tuple)) else value)
                for value in data
            ]
        except (TypeError, ValueError, IndexError) as exc:
            raise TypeError("时间表必须是一维数据或单列数据") from exc
