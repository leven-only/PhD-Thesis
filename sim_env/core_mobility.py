"""车辆移动执行模块，用于根据路径计划和路网状态推动车辆行驶。"""

from dataclasses import dataclass
from typing import Any, Optional

from core_road_network import RoadNetwork
from core_vehicle import Vehicle, VehicleManager, VehicleStatus


@dataclass
class MobilityPlan:
    """单辆车当前正在执行的路径计划。"""

    vehicle_id: str
    path: list[str]
    next_node_index: int = 1
    current_edge_remaining_km: Optional[float] = None


@dataclass
class MovementResult:
    """单辆车在一个时间步内的移动结果。"""

    distance_km: float = 0.0
    travel_time: float = 0.0
    reached_node_id: Optional[str] = None
    status: Optional[VehicleStatus] = None
    remove_plan: bool = False


class MobilityManager:
    """车辆移动执行器。

    该类只负责执行路径，不负责决定路径为什么这样选。
    路径必须由外部算法或调用方传入。
    """

    def __init__(
        self,
        road_network: RoadNetwork,
        vehicle_manager: VehicleManager,
    ) -> None:
        self.road_network = road_network    # 路网对象
        self.vehicle_manager = vehicle_manager  # 汽车对象管理
        self.active_plans: dict[str, MobilityPlan] = {} # 当前所有汽车的移动计划，key为汽车id，value为移动计划对象

    # 清空当前路径计划。
    def reset(self) -> None:
        self.active_plans = {}

    # 推进所有正在执行路径的车辆。
    def step(self, time_step: float, current_time: float, action: Optional[Any] = None) -> None:
        self._apply_action(action)  # 给每个汽车
        self._ensure_active_plans()

        for vehicle_id in list(self.active_plans.keys()):
            self._move_vehicle(vehicle_id, time_step)

    # 返回移动执行层状态
    def get_state(self) -> dict[str, Any]:
        return {
            "active_plan_count": len(self.active_plans),
            "active_plans": {
                vehicle_id: {
                    "path": plan.path,
                    "next_node_index": plan.next_node_index,
                    "current_edge_remaining_km": plan.current_edge_remaining_km,
                }
                for vehicle_id, plan in self.active_plans.items()
            },
        }

    # 给每个汽车分配移动路径
    def _apply_action(self, action: Optional[Any]) -> None:
        if not isinstance(action, dict):
            return

        paths = action.get("vehicle_paths")

        if not isinstance(paths, dict):
            return

        """
        action = {
            "vehicle_paths": {
                "vehicle_001": ["node_01", "node_02", "node_06", "node_10"]
            }
        }
        """
        for vehicle_id, path in paths.items():
            self._set_path(vehicle_id, path)

    # 检查所有应该移动的车，是否已经有可执行的路径计划。
    def _ensure_active_plans(self) -> None:
        for vehicle_id, vehicle in self.vehicle_manager.vehicles.items():
            if not vehicle.can_move():  # 汽车能移动返回true，否则false
                continue

            # 跳过已经到达终点的汽车
            if vehicle.current_node_id == vehicle.destination_node_id:
                vehicle.update_state(status=VehicleStatus.FINISHED)
                continue

            # 当前汽车没有计划，报错
            if vehicle_id not in self.active_plans:
                raise ValueError(f"车辆缺少路径计划: {vehicle_id}")

    def _move_vehicle(self, vehicle_id: str, time_step: float) -> None:
        vehicle = self.vehicle_manager.get_vehicle(vehicle_id)  # 汽车对象
        plan = self.active_plans[vehicle_id]    # 移动计划对象

        # 移动函数只负责执行移动流程：计算移动结果，再交给车辆统一更新状态。
        result = self._calculate_movement_result(vehicle, plan, time_step)

        # 更新汽车自身状态
        vehicle.apply_movement_result(
            distance_km=result.distance_km,
            travel_time=result.travel_time,
            current_node_id=result.reached_node_id,
            status=result.status,
        )

        # 删除计划
        if result.remove_plan:
            self.active_plans.pop(vehicle_id, None)

    # 计算 汽车 移动
    def _calculate_movement_result(
        self,
        vehicle: Vehicle,
        plan: MobilityPlan,
        time_step: float,
    ) -> MovementResult:
        """根据路径计划和路网状态，计算一个时间步内的车辆移动结果。"""
        remaining_time = max(time_step, 0.0)    # 当前行动计划剩余时间
        remaining_energy_distance = vehicle.available_distance_km() # 剩余可行驶距离
        result = MovementResult()

        while remaining_time > 0 and vehicle.can_move():    # 只要还有剩余时间 并且 汽车可以移动
            if plan.next_node_index >= len(plan.path):  # 路径走完了
                result.status = VehicleStatus.FINISHED
                result.remove_plan = True
                break

            # 找到当前要行驶的路段。
            current_node = plan.path[plan.next_node_index - 1]
            next_node = plan.path[plan.next_node_index]
            edge = self.road_network.get_edge_between(current_node, next_node)

            if plan.current_edge_remaining_km is None:
                plan.current_edge_remaining_km = edge.length_km

            # 本轮可行驶距离同时受到时间、剩余路段长度和剩余电量限制。
            speed_km_per_second = edge.current_speed_kph / 3600
            distance_by_time = speed_km_per_second * remaining_time
            travel_distance = min(
                plan.current_edge_remaining_km,
                distance_by_time,
                remaining_energy_distance,
            )

            if travel_distance <= 0:
                result.status = VehicleStatus.FAILED
                result.remove_plan = True
                break

            # 只累计临时移动结果，不在这里直接改车辆状态。
            travel_time = travel_distance / speed_km_per_second
            result.distance_km += travel_distance
            result.travel_time += travel_time
            remaining_time -= travel_time
            remaining_energy_distance -= travel_distance
            plan.current_edge_remaining_km -= travel_distance

            if remaining_energy_distance <= 1e-9:
                result.status = VehicleStatus.FAILED
                result.remove_plan = True
                break

            # 当前时间步不够走完整条边，车辆停留在边上，计划下次继续执行。
            if plan.current_edge_remaining_km > 1e-9:
                break

            # 当前边已经走完，车辆到达下一个节点，计划推进到下一段路。
            result.reached_node_id = next_node
            plan.next_node_index += 1
            plan.current_edge_remaining_km = None

        if result.reached_node_id == vehicle.destination_node_id:
            result.status = VehicleStatus.FINISHED
            result.remove_plan = True

        return result

    # 给车辆设置一条待执行路径，建立移动计划对象。
    def _set_path(self, vehicle_id: str, path: list[str]) -> None:
        vehicle = self.vehicle_manager.get_vehicle(vehicle_id)  # 获取 vehicle 对象

        if not path:
            raise ValueError("路径不能为空")

        if vehicle.current_node_id != path[0]:
            raise ValueError(
                f"路径起点 {path[0]} 与车辆当前位置 {vehicle.current_node_id} 不一致"
            )

        if len(path) == 1:  # 路段节点即汽车终点
            vehicle.update_state(status=VehicleStatus.FINISHED)
            self.active_plans.pop(vehicle_id, None)
            return

        # 建立移动对象
        self.active_plans[vehicle_id] = MobilityPlan(
            vehicle_id=vehicle_id,
            path=list(path),
        )

        vehicle.update_state(status=VehicleStatus.DRIVING)