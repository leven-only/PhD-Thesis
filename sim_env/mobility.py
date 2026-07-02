"""车辆移动执行模块，用于根据路径计划和路网状态推动车辆行驶。"""

from dataclasses import dataclass
from typing import Any, Optional

from sim_env.road_network import RoadNetwork
from sim_env.vehicle import Vehicle, VehicleManager, VehicleStatus


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
    路径可以由外部算法传入；测试阶段也可以启用最短路径 baseline。
    """

    def __init__(
        self,
        road_network: RoadNetwork,
        vehicle_manager: VehicleManager,
        auto_plan_shortest_path: bool = True,
        path_weight: str = "time",
    ) -> None:
        self.road_network = road_network
        self.vehicle_manager = vehicle_manager
        self.auto_plan_shortest_path = auto_plan_shortest_path
        self.path_weight = path_weight
        self.active_plans: dict[str, MobilityPlan] = {}

    def reset(self) -> None:
        """清空当前路径计划，并按需生成默认最短路径计划。"""
        self.active_plans = {}

        if self.auto_plan_shortest_path:
            self.plan_all_shortest_paths()

    def step(self, time_step: float, current_time: float, action: Optional[Any] = None) -> None:
        """推进所有正在执行路径的车辆。"""
        self._apply_action(action)

        if self.auto_plan_shortest_path:
            self.plan_missing_shortest_paths()

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
            vehicle.mark_finished()
            self.active_plans.pop(vehicle_id, None)
            return

        self.active_plans[vehicle_id] = MobilityPlan(
            vehicle_id=vehicle_id,
            path=list(path),
        )
        vehicle.set_status(VehicleStatus.DRIVING)

    def plan_shortest_path_for_vehicle(self, vehicle_id: str) -> None:
        """给单辆车生成到目的地的当前最短路径。"""
        vehicle = self.vehicle_manager.get_vehicle(vehicle_id)

        if vehicle.current_node_id is None:
            raise ValueError(f"车辆当前位置为空: {vehicle_id}")

        path = self.road_network.shortest_path(
            vehicle.current_node_id,
            vehicle.destination_node_id,
            weight=self.path_weight,
        )
        self.set_path(vehicle_id, path)

    def plan_all_shortest_paths(self) -> None:
        """给所有未完成车辆生成最短路径。"""
        for vehicle_id in self.vehicle_manager.vehicles:
            self._try_plan_shortest_path(vehicle_id)

    def plan_missing_shortest_paths(self) -> None:
        """给没有路径计划的可行驶车辆补充最短路径。"""
        for vehicle_id in self.vehicle_manager.vehicles:
            if vehicle_id in self.active_plans:
                continue

            self._try_plan_shortest_path(vehicle_id)

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

    def _try_plan_shortest_path(self, vehicle_id: str) -> None:
        vehicle = self.vehicle_manager.get_vehicle(vehicle_id)

        if vehicle.status in (
            VehicleStatus.FINISHED,
            VehicleStatus.FAILED,
            VehicleStatus.CHARGING,
            VehicleStatus.QUEUEING,
        ):
            return

        if vehicle.current_node_id == vehicle.destination_node_id:
            vehicle.mark_finished()
            return

        self.plan_shortest_path_for_vehicle(vehicle_id)

    def _move_vehicle(self, vehicle_id: str, time_step: float) -> None:
        vehicle = self.vehicle_manager.get_vehicle(vehicle_id)
        plan = self.active_plans[vehicle_id]

        remaining_time = max(time_step, 0.0)

        while remaining_time > 0 and vehicle.status == VehicleStatus.DRIVING:
            if plan.next_node_index >= len(plan.path):
                vehicle.mark_finished()
                self.active_plans.pop(vehicle_id, None)
                return

            current_node = plan.path[plan.next_node_index - 1]
            next_node = plan.path[plan.next_node_index]
            edge = self.road_network.get_edge_between(current_node, next_node)

            if plan.current_edge_remaining_km is None:
                plan.current_edge_remaining_km = edge.length_km

            speed_km_per_second = edge.current_speed_kph / 3600
            distance_by_time = speed_km_per_second * remaining_time
            distance_by_energy = self._available_distance_km(vehicle)
            travel_distance = min(
                plan.current_edge_remaining_km,
                distance_by_time,
                distance_by_energy,
            )

            if travel_distance <= 0:
                vehicle.set_status(VehicleStatus.FAILED)
                self.active_plans.pop(vehicle_id, None)
                return

            travel_time = travel_distance / speed_km_per_second
            vehicle.record_travel(travel_distance, travel_time)
            remaining_time -= travel_time
            plan.current_edge_remaining_km -= travel_distance

            if vehicle.status == VehicleStatus.FAILED:
                self.active_plans.pop(vehicle_id, None)
                return

            if plan.current_edge_remaining_km > 1e-9:
                return

            vehicle.move_to_node(next_node)
            plan.next_node_index += 1
            plan.current_edge_remaining_km = None

        if vehicle.current_node_id == vehicle.destination_node_id:
            vehicle.mark_finished()
            self.active_plans.pop(vehicle_id, None)

    def _available_distance_km(self, vehicle: Vehicle) -> float:
        if vehicle.energy_consumption_kwh_per_km <= 0:
            return float("inf")

        available_energy_kwh = vehicle.soc * vehicle.battery_capacity_kwh
        return available_energy_kwh / vehicle.energy_consumption_kwh_per_km
