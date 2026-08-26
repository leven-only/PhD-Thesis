"""路网组件：负责维护道路拓扑，并根据仿真时间刷新道路速度。

角色说明：RoadNetwork 属于"世界状态"（world state），不是反应式组件——
它只根据 current_time 更新自身，不接收/产生 events，因此不能套用
EVChargingEnv 给反应式组件（Vehicle/Station/Mobility 等）用的统一
`step(time_step, current_time, action, events)` 调用方式；应该由 env
在每个 tick 显式调用 `step(current_time)`。

不做输入校验：调用方需要自己保证 matrix/speed_matrix/speed_timetable
格式正确（方阵、非负、对角线为 0、速度行数跟非零元素个数对应、时间表
严格递增），构造时不会检查、也不会转换数据类型。
"""

from bisect import bisect_right
from typing import Any

import networkx as nx


class RoadNetwork:
    """由邻接矩阵和速度数据构成的固定拓扑路网。

    ``matrix`` 的非零值表示道路长度（km）。矩阵按从上到下、从左到右扫描；
    每个非零元素依次对应 ``speed_matrix`` 的一列，``speed_matrix`` 的每一行
    表示一个速度快照。矩阵的行列下标直接作为节点 ID。
    """

    def __init__(
        self,
        matrix: Any,
        speed_matrix: Any,
        speed_timetable: Any,
        start_time: float,
    ) -> None:
        self.matrix = matrix
        self.speed_matrix = speed_matrix
        self.speed_timetable = speed_timetable
        self.start_time = start_time

        self.node_count = len(self.matrix)
        self._initialize_network()
        self.reset()

    # ------------------------------------------------------------------
    # 世界状态接口：reset / step / get_state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """把所有道路速度恢复到 start_time 对应的速度快照。"""
        self._apply_speed_snapshot(self._snapshot_index_at(self.start_time))

    def step(self, current_time: float) -> None:
        """根据当前仿真时间刷新道路速度；如果快照没变则什么都不做。"""
        snapshot_index = self._snapshot_index_at(current_time)
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
        """返回两个节点之间道路的属性字典，供其他组件（如 mobility）直接使用。

        返回的是图中边属性的直接引用，只建议读取；速度更新统一由 step() 完成。
        """
        try:
            return self.graph.edges[node_u, node_v]
        except (KeyError, nx.NetworkXError) as exc:
            raise ValueError(f"节点之间不存在道路: {node_u} -> {node_v}") from exc

    # ------------------------------------------------------------------
    # 内部实现（下划线开头，外部代码不应依赖）
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
        """用指定下标的速度快照更新全部道路的速度和通行时间。"""
        speeds = self.speed_matrix[snapshot_index]
        for (row, column), speed in zip(self._edge_positions, speeds):
            edge_data = self.graph.edges[row, column]
            edge_data["speed_kph"] = speed
            edge_data["travel_time_seconds"] = edge_data["length_km"] / max(speed, 1e-6) * 3600
        self.current_speed_snapshot_index = snapshot_index

    def _snapshot_index_at(self, current_time: float) -> int:
        """返回不晚于 current_time 的最近速度快照下标。"""
        return max(bisect_right(self.speed_timetable, current_time) - 1, 0)
