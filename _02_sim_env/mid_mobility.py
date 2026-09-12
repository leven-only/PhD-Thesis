"""MobilityManager：负责推进车辆沿着出行方案（节点路径）移动。"""

from bisect import bisect_right
from copy import deepcopy
from typing import Optional

from _02_sim_env.base_road_network import RoadNetwork
from _02_sim_env.base_vehicle import Vehicle


class MobilityManager:
    def __init__(self, road_network: RoadNetwork, vehicle: Vehicle) -> None:
        """绑定road_network和vehicle，方案相关的属性先初始化为空，等外部再下发方案。"""
        self.road_network = road_network  # 路网组件，查边长/限速用
        self.vehicle = vehicle            # 车辆对象，读写车辆的位置/电量/状态用

        self.path: Optional[list[int]] = None  # 出行方案：节点id组成的路径，没有方案时是None
        self.milestones: list[float] = []      # 里程碑表：milestones[i]是从起点走到path[i]的累计公里数
        self.path_progress: float = 0.0        # 沿着path已经累计走了多少公里

        # 构造完成时的初始值快照，供reset()还原用
        self.initial_state = {
            "path": self.path,
            "milestones": self.milestones,
            "path_progress": self.path_progress,
        }

    def reset(self) -> None:
        """把方案相关的属性都还原成构造完成时的初始值（清空当前方案）。"""
        for name, value in self.initial_state.items():
            setattr(self, name, deepcopy(value))

    def assign_plan(self, path: list[int]) -> None:
        """接收一个新的出行方案：先reset()清空上一个方案的残留，再算出这条路径的里程碑表。
        供Env复用同一个MobilityManager时调用（不用每次都new一个新对象）。
        """
        self.reset()
        if len(path) < 2:
            raise ValueError("路径至少要有两个节点（当前节点 + 下一个目标节点）")
        if self.vehicle.current_node_id != path[0]:
            raise ValueError(f"路径起点 {path[0]} 与车辆当前位置 {self.vehicle.current_node_id} 不一致")

        self.path = list(path)
        self.milestones = self._build_milestones(self.path)  # 提前把每个节点的累计里程算好
        self.path_progress = 0.0                              # 从起点出发，目前累计走了0公里

    def step(self):
        """推进车辆走完当前所在的这一条边，到达下一个节点为止——不设时间预算
        上限，这条边多长就走多久（车辆在路上的时候不看环境变化，所以没必要
        卡在一个固定的time_step上，只在到达节点这一刻才需要停下来，把控制权
        交还给上层Env）。

        返回(finished_flag, time_used)：
        finished_flag=1：这条边刚好是路径的最后一段，走完这一步整条方案就
        走到头了（不一定是车辆真正的终点，是不是真终点由上层判断），已经把
        自己reset()清空。
        finished_flag=0：到了下一个节点，但方案还没走到头，需要外部再调用
        一次step()才能接着走下一条边。
        time_used是走这条边实际用掉的时间（分钟）。
        """
        # 移动前先检查有没有方案：没有就说明step()前忘了调用assign_plan()
        if self.path is None:
            raise RuntimeError("没有方案，请先调用assign_plan()初始化path等参数，再调用step()")

        total_length_km = self.milestones[-1]  # 整条方案一共要走多少公里

        # 根据目前累计走了多少公里，查出正在哪一条边上
        segment_index = self._locate_segment_index()
        node_from = self.path[segment_index]
        node_to = self.path[segment_index + 1]
        edge = self.road_network.get_edge_between(node_from, node_to)

        distance_km = self.milestones[segment_index + 1] - self.path_progress  # 这条边还剩多少公里，这一步全部走完

        # 全局时间单位是分钟，把限速(公里/小时)换算成公里/分钟
        speed_km_per_minute = max(edge["speed_kph"], 1e-6) / 60.0
        travel_time = distance_km / speed_km_per_minute  # 走完这条边实际用掉的时间

        self.path_progress = self.path_progress + distance_km

        # 走完这条边，是不是也刚好把整条路径走到头了
        finished_flag = 1 if self.path_progress >= total_length_km - 1e-9 else 0

        current_node_id, next_node_id, edge_progress_km = self._current_position()

        self.vehicle.step(
            action="move",
            params={
                "distance_km": distance_km,
                "travel_time": travel_time,
                "current_node_id": current_node_id,
                "next_node_id": next_node_id,
                "edge_progress_km": edge_progress_km,
            },
        )

        time_used = travel_time  # 走这条边实际用掉的时间，就是这一步的time_used

        if finished_flag == 1:
            self.reset()  # 方案走完了，清空自己，等外部下一次assign_plan()

        return finished_flag, time_used

    def _build_milestones(self, path: list[int]) -> list[float]:
        """把路径上每段边的长度依次累加：milestones[i]是从path[0]走到path[i]的累计公里数。"""
        milestones = [0.0]  # 起点(path[0])的累计里程是0
        for index in range(1, len(path)):
            edge = self.road_network.get_edge_between(path[index - 1], path[index])
            milestones.append(milestones[index - 1] + edge["length_km"])
        return milestones

    def _locate_segment_index(self) -> int:
        """根据path_progress在milestones里查出当前在哪一条边上，返回这条边起点在path里的下标。"""
        index = bisect_right(self.milestones, self.path_progress) - 1
        index = max(index, 0)                    # path_progress=0时停在第一条边上
        index = min(index, len(self.path) - 2)   # 不越界（最后一条边的起点下标是len(path)-2）
        return index

    def _current_position(self):
        """算出车辆当前在哪两个节点之间、这条边已经走了多远。"""
        total_length_km = self.milestones[-1]
        if self.path_progress >= total_length_km:
            last_node = self.path[-1]
            return last_node, last_node, 0.0

        segment_index = self._locate_segment_index()
        node_from = self.path[segment_index]
        node_to = self.path[segment_index + 1]
        edge_progress_km = self.path_progress - self.milestones[segment_index]
        return node_from, node_to, edge_progress_km
