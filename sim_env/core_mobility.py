"""移动组件，负责执行路径计划并产生车辆移动事件。"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

from sim_env.core_road_network import RoadNetwork
from sim_env.core_vehicle import (
    Vehicle,
    VehicleEvent,
    VehicleManager,
    VehicleStatus,
)


@dataclass
class MobilityPlan:
    """一辆车正在执行的路径计划。"""

    vehicle_id: str
    path: list[int]
    next_node_index: int = 1
    current_edge_remaining_km: Optional[float] = None


@dataclass
class MovementResult:
    """一辆车在一个时间步中的移动计算结果。"""

    distance_km: float = 0.0
    travel_time: float = 0.0
    reached_node_id: Optional[int] = None
    status: Optional[VehicleStatus] = None
    remove_plan: bool = False


class MobilityManager:
    """只更新路径计划，并把移动结果发布给 VehicleManager。"""

    def __init__(
        self,
        road_network: RoadNetwork,
        vehicle_manager: VehicleManager,
    ) -> None:
        self.road_network = road_network
        self.vehicle_manager = vehicle_manager
        self._active_plans: dict[str, MobilityPlan] = {}

    def reset(self) -> None:
        """清空全部路径计划。"""
        self._active_plans = {}

    def step(
        self,
        time_step: float,
        current_time: float,
        action: Optional[Any] = None,
        events: Optional[list[Any]] = None,
    ) -> list[Any]:
        """应用路径动作、推进计划并返回移动事件。"""
        produced_events = self._apply_path_actions(action)

        for vehicle_id in list(self._active_plans):
            produced_events.append(
                self._advance_plan(vehicle_id, time_step)
            )

        return produced_events

    def get_state(self) -> dict[str, Any]:
        """返回当前路径执行状态。"""
        return {
            "active_plan_count": len(self._active_plans),
            "active_plans": {
                vehicle_id: {
                    "path": deepcopy(plan.path),
                    "next_node_index": plan.next_node_index,
                    "current_edge_remaining_km": plan.current_edge_remaining_km,
                }
                for vehicle_id, plan in self._active_plans.items()
            },
        }

    def _apply_path_actions(self, action: Optional[Any]) -> list[Any]:
        if not isinstance(action, dict):
            return []

        paths = action.get("vehicle_paths")
        if not isinstance(paths, dict):
            return []

        produced_events: list[Any] = []
        for vehicle_id, path in paths.items():
            event = self._set_path(vehicle_id, path)
            if event is not None:
                produced_events.append(event)

        return produced_events

    def _set_path(
        self,
        vehicle_id: str,
        path: list[int],
    ) -> Optional[VehicleEvent]:
        vehicle = self.vehicle_manager.get_vehicle(vehicle_id)
        if not path:
            raise ValueError("路径不能为空")
        if vehicle.current_node_id != path[0]:
            raise ValueError(
                f"路径起点 {path[0]} 与车辆当前位置 "
                f"{vehicle.current_node_id} 不一致"
            )

        if len(path) == 1:
            if path[0] != vehicle.destination_node_id:
                raise ValueError("单节点路径只能用于已经到达终点的车辆")

            self._active_plans.pop(vehicle_id, None)
            return VehicleEvent(
                vehicle_id=vehicle_id,
                event_type="status",
                data={"status": VehicleStatus.FINISHED},
            )

        self._active_plans[vehicle_id] = MobilityPlan(
            vehicle_id=vehicle_id,
            path=list(path),
        )
        return None

    def _advance_plan(
        self,
        vehicle_id: str,
        time_step: float,
    ) -> VehicleEvent:
        vehicle = self.vehicle_manager.get_vehicle(vehicle_id)
        plan = self._active_plans[vehicle_id]
        result = self._calculate_movement(vehicle, plan, time_step)

        if result.remove_plan:
            self._active_plans.pop(vehicle_id, None)

        return VehicleEvent(
            vehicle_id=vehicle_id,
            event_type="movement",
            data={
                "distance_km": result.distance_km,
                "travel_time": result.travel_time,
                "current_node_id": result.reached_node_id,
                "status": result.status,
            },
        )

    def _calculate_movement(
        self,
        vehicle: Vehicle,
        plan: MobilityPlan,
        time_step: float,
    ) -> MovementResult:
        remaining_time = max(float(time_step), 0.0)
        remaining_energy_distance = vehicle.available_distance_km()
        result = MovementResult()

        while remaining_time > 0 and self._can_move(vehicle):
            if plan.next_node_index >= len(plan.path):
                result.status = VehicleStatus.FINISHED
                result.remove_plan = True
                break

            current_node = plan.path[plan.next_node_index - 1]
            next_node = plan.path[plan.next_node_index]
            edge = self.road_network.get_edge_between(current_node, next_node)

            if plan.current_edge_remaining_km is None:
                plan.current_edge_remaining_km = edge["length_km"]

            speed_km_per_second = max(edge["speed_kph"], 1e-6) / 3600
            travel_distance = min(
                plan.current_edge_remaining_km,
                speed_km_per_second * remaining_time,
                remaining_energy_distance,
            )

            if travel_distance <= 0:
                result.status = VehicleStatus.FAILED
                result.remove_plan = True
                break

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

            if plan.current_edge_remaining_km > 1e-9:
                break

            result.reached_node_id = next_node
            plan.next_node_index += 1
            plan.current_edge_remaining_km = None

        if result.reached_node_id == vehicle.destination_node_id:
            result.status = VehicleStatus.FINISHED
            result.remove_plan = True
        elif result.distance_km > 0 and result.status is None:
            result.status = VehicleStatus.DRIVING

        return result

    @staticmethod
    def _can_move(vehicle: Vehicle) -> bool:
        return vehicle.status not in (
            VehicleStatus.FINISHED,
            VehicleStatus.FAILED,
            VehicleStatus.CHARGING,
            VehicleStatus.QUEUEING,
        )
