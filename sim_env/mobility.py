"""车辆移动执行模块，用于根据路径计划和路网状态推动车辆行驶。"""

from dataclasses import dataclass
from typing import Any, Optional

from sim_env.road_network import RoadNetwork
from sim_env.vehicle import VehicleManager, VehicleStatus


@dataclass
class MobilityPlan:
    """单辆车当前正在执行的路径计划。"""

    vehicle_id: str
    path: list[str]
    next_node_index: int = 1
    current_edge_remaining_km: Optional[float] = None


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
        self.active_plans: dict[str, MobilityPlan] = {}

    def reset(self) -> None:
        """清空当前路径计划。"""
        self.active_plans = {}

    def step(self, time_step: float, current_time: float, action: Optional[Any] = None) -> None:
        """推进所有正在执行路径的车辆。"""
        self._apply_action(action)
        self._ensure_active_plans()

        for vehicle_id in list(self.active_plans.keys()):
            self._move_vehicle(vehicle_id, time_step)

    def set_path(self, vehicle_id: str, path: list[str]) -> None:
        """给车辆设置一条待执行路径。"""
        vehicle = self.vehicle_manager.get_vehicle(vehicle_id)

        if not path:
            raise ValueError("路径不能为空")

        if vehicle.current_node_id != path[0]:
            raise ValueError(
                f"路径起点 {path[0]} 与车辆当前位置 {vehicle.current_node_id} 不一致"
            )

        if len(path) == 1:
            vehicle.update_state(status=VehicleStatus.FINISHED)
            self.active_plans.pop(vehicle_id, None)
            return

        self.active_plans[vehicle_id] = MobilityPlan(
            vehicle_id=vehicle_id,
            path=list(path),
        )
        vehicle.update_state(status=VehicleStatus.DRIVING)

    def get_state(self) -> dict[str, Any]:
        """返回移动执行层状态。"""
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

    def _apply_action(self, action: Optional[Any]) -> None:
        """应用外部算法传入的路径计划。"""
        if not isinstance(action, dict):
            return

        paths = action.get("vehicle_paths")

        if not isinstance(paths, dict):
            return

        for vehicle_id, path in paths.items():
            self.set_path(vehicle_id, path)

    def _ensure_active_plans(self) -> None:
        """确保所有可行驶车辆都有路径计划。"""
        for vehicle_id, vehicle in self.vehicle_manager.vehicles.items():
            if not vehicle.can_move():
                continue

            if vehicle.current_node_id == vehicle.destination_node_id:
                vehicle.update_state(status=VehicleStatus.FINISHED)
                continue

            if vehicle_id not in self.active_plans:
                raise ValueError(f"车辆缺少路径计划: {vehicle_id}")

    def _move_vehicle(self, vehicle_id: str, time_step: float) -> None:
        vehicle = self.vehicle_manager.get_vehicle(vehicle_id)
        plan = self.active_plans[vehicle_id]

        remaining_time = max(time_step, 0.0)
        remaining_energy_distance = vehicle.available_distance_km()

        while remaining_time > 0 and vehicle.can_move():
            if plan.next_node_index >= len(plan.path):
                vehicle.update_state(status=VehicleStatus.FINISHED)
                self.active_plans.pop(vehicle_id, None)
                return

            current_node = plan.path[plan.next_node_index - 1]
            next_node = plan.path[plan.next_node_index]
            edge = self.road_network.get_edge_between(current_node, next_node)

            if plan.current_edge_remaining_km is None:
                plan.current_edge_remaining_km = edge.length_km

            speed_km_per_second = edge.current_speed_kph / 3600
            distance_by_time = speed_km_per_second * remaining_time
            travel_distance = min(
                plan.current_edge_remaining_km,
                distance_by_time,
                remaining_energy_distance,
            )

            if travel_distance <= 0:
                vehicle.update_state(status=VehicleStatus.FAILED)
                self.active_plans.pop(vehicle_id, None)
                return

            travel_time = travel_distance / speed_km_per_second
            self._update_vehicle_travel(vehicle, travel_distance, travel_time)
            remaining_time -= travel_time
            remaining_energy_distance -= travel_distance
            plan.current_edge_remaining_km -= travel_distance

            if vehicle.is_failed():
                self.active_plans.pop(vehicle_id, None)
                return

            if plan.current_edge_remaining_km > 1e-9:
                return

            vehicle.update_state(current_node_id=next_node)
            plan.next_node_index += 1
            plan.current_edge_remaining_km = None

        if vehicle.current_node_id == vehicle.destination_node_id:
            vehicle.update_state(status=VehicleStatus.FINISHED)
            self.active_plans.pop(vehicle_id, None)

    def _update_vehicle_travel(self, vehicle, travel_distance: float, travel_time: float) -> None:
        energy_used = max(travel_distance, 0.0) * vehicle.energy_consumption_kwh_per_km
        soc_drop = energy_used / vehicle.battery_capacity_kwh
        new_soc = max(0.0, vehicle.soc - soc_drop)
        new_status = VehicleStatus.FAILED if new_soc <= 0.0 else vehicle.status

        vehicle.update_state(
            soc=new_soc,
            status=new_status,
            total_distance_km=vehicle.total_distance_km + max(travel_distance, 0.0),
            total_energy_used_kwh=vehicle.total_energy_used_kwh + energy_used,
            total_travel_time=vehicle.total_travel_time + max(travel_time, 0.0),
        )
