"""MobilityManager：负责推进车辆沿着出行方案（节点路径）移动。"""

from bisect import bisect_right
from copy import deepcopy
from typing import Optional

from sim_env.base_road_network import RoadNetwork
from sim_env.base_vehicle import Vehicle, VehicleStatus


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

    def step(self, available_time: float):
        """在available_time给出的时间预算内推进车辆移动，时间到了就返回；由上层
        （Env）拿返回值去更新时钟、刷新环境，需要的话再调用step()接着走。

        available_time是"距离下一次环境更新时间点还有多久"，由上层算好传进来
        （不一定等于仿真固定的time_step——如果这一路上已经先用掉一部分时间，
        剩下能走的就会比time_step短）；用它限制这一步能走多远，是为了让车辆的
        移动正好卡在环境下一次更新的时间点上，不会在旧的环境状态下走了很久才更新。

        返回(finished_flag, time_used)：
        finished_flag=0：available_time用完了，方案还没走完（自然终止，刚好卡到
        了下一个环境更新的时间点），time_used就是整个available_time；下次再调用
        step()会接着这次的断点继续走。
        finished_flag=1：方案在这一步之内就走到头了（不一定是车辆真正的终点，是
        不是真终点由上层判断），已经把自己reset()清空；time_used是这一步实际用
        掉的时间（小于等于available_time），没用完的部分可以被外部当成这一轮还
        剩下的可用时间去做别的事。
        """
        # 移动前先检查有没有方案：没有就说明step()前忘了调用assign_plan()
        if self.path is None:
            raise RuntimeError("没有方案，请先调用assign_plan()初始化path等参数，再调用step()")

        total_length_km = self.milestones[-1]  # 整条方案一共要走多少公里
        remaining_time = available_time        # 这一步还剩多少时间预算没用掉
        distance_km = 0.0                      # 这一步总共走了多少公里
        travel_time = 0.0                      # 这一步总共花了多少时间
        finished_flag = 0                      # 先假设是"预算用完"，方案真走到头了再改成1

        # 一条边一条边地往前走，直到时间预算用完，或者方案走到头
        while remaining_time > 0:
            if self.path_progress >= total_length_km:
                finished_flag = 1
                break

            # 根据目前累计走了多少公里，查出正在哪一条边上
            segment_index = self._locate_segment_index()
            node_from = self.path[segment_index]
            node_to = self.path[segment_index + 1]
            edge = self.road_network.get_edge_between(node_from, node_to)

            edge_remaining_km = self.milestones[segment_index + 1] - self.path_progress  # 这条边还剩多少公里没走完

            # 全局时间单位是分钟，把限速(公里/小时)换算成公里/分钟
            speed_km_per_minute = max(edge["speed_kph"], 1e-6) / 60.0

            # 这一小段能走多远：取"这条边剩多远"和"这些时间能走多远"两者中更小的
            step_distance = min(edge_remaining_km, speed_km_per_minute * remaining_time)
            step_time = step_distance / speed_km_per_minute

            distance_km = distance_km + step_distance
            travel_time = travel_time + step_time
            remaining_time = remaining_time - step_time
            self.path_progress = self.path_progress + step_distance

            if step_distance < edge_remaining_km - 1e-9:
                break  # 这条边没走完，说明这一步的时间预算已经用光了
            # 否则这条边刚好走完了，继续下一轮循环，用剩下的时间接着走下一条边

        current_node_id, next_node_id, edge_progress_km = self._current_position()

        # 这一步只要真的动了，就报DRIVING；是不是真的到终点，交给上层（Env）判断
        if distance_km > 0:
            move_status = VehicleStatus.DRIVING
        else:
            move_status = None

        self.vehicle.step(
            action="move",
            params={
                "distance_km": distance_km,
                "travel_time": travel_time,
                "current_node_id": current_node_id,
                "next_node_id": next_node_id,
                "edge_progress_km": edge_progress_km,
                "status": move_status,
            },
        )

        if finished_flag == 1:
            time_used = travel_time  # 真实耗时，可能小于available_time
            self.reset()             # 方案走完了，清空自己，等外部下一次assign_plan()
        else:
            time_used = available_time  # 预算全部用掉

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
