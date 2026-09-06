"""移动组件，负责执行路径计划、算出车辆移动结果，并直接写回车辆。"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

from sim_env.base_road_network import RoadNetwork
from sim_env.base_vehicle import Vehicle, VehicleManager, VehicleStatus


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
    current_node_id: Optional[int] = None  # 这一步结束时，最后完全到达的节点
    next_node_id: Optional[int] = None     # 这一步结束时，正朝向的下一个节点（可能还没到）
    edge_progress_km: float = 0.0          # 在current_node_id->next_node_id这条边上已经走了多远
    status: Optional[VehicleStatus] = None
    remove_plan: bool = False


class MobilityManager:
    """只负责计算"这一步车辆移动了多少"，并把结果直接应用到车辆上。

    跟ChargingStation/RoadNetwork这类"世界状态"组件不同，也跟VehicleManager/
    StationManager这类"集合管理器"不同——这是一个横跨RoadNetwork和Vehicle两个
    基础组件的领域协调器：结合路网当前的边速度、车辆当前的位置和剩余电量，
    算出这一步车辆实际能走多远，再直接调用VehicleManager.step()把结果写回车辆，
    不再像之前那样包成VehicleEvent对象、指望外部组件另外去应用。

    读车辆当前状态用的是vehicle_manager.get_vehicle()（深拷贝，只读、不会改到
    真实对象），写回用的是vehicle_manager.step(actions=...)（会真的改到内部
    真实车辆），这两条路径分工明确、不会混用。
    """

    def __init__(
        self,
        road_network: RoadNetwork,
        vehicle_manager: VehicleManager,
    ) -> None:
        self.road_network = road_network
        self.vehicle_manager = vehicle_manager
        self._active_plans: dict[str, MobilityPlan] = {}

    # ------------------------------------------------------------------
    # 反应式组件接口：reset / step / get_state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """清空全部路径计划。"""
        self._active_plans = {}

    def step(self, time_step: float, action: Optional[Any] = None) -> None:
        """应用路径动作、推进各车辆的计划，把移动结果直接写回车辆。

        不看current_time：这一步用到的"路网当前边速度"这类跟时间有关的信息，
        应该是外部（Env）在调用这里之前先调用过road_network.step(current_time)
        刷新好的，Mobility自己只管"给定这段时间(time_step)，车辆能走多远"，不
        需要再关心现在几点，这跟Vehicle.step()不看时间是同一个道理。
        """
        actions: dict[str, dict[str, Any]] = {}

        self._apply_path_actions(action, actions)

        for vehicle_id in list(self._active_plans):
            self._advance_plan(vehicle_id, time_step, actions)

        if actions:
            self.vehicle_manager.step(actions=actions)

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

    # ------------------------------------------------------------------
    # 内部实现（下划线开头，外部代码不应依赖）
    # ------------------------------------------------------------------

    def _apply_path_actions(
        self,
        action: Optional[Any],
        actions: dict[str, dict[str, Any]],
    ) -> None:
        if not isinstance(action, dict):
            return

        paths = action.get("vehicle_paths")
        if not isinstance(paths, dict):
            return

        for vehicle_id, path in paths.items():
            self._set_path(vehicle_id, path, actions)

    def _set_path(
        self,
        vehicle_id: str,
        path: list[int],
        actions: dict[str, dict[str, Any]],
    ) -> None:
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
            actions[vehicle_id] = {
                "action": "set_status",
                "params": {"status": VehicleStatus.FINISHED},
            }
            return

        self._active_plans[vehicle_id] = MobilityPlan(
            vehicle_id=vehicle_id,
            path=list(path),
        )

    def _advance_plan(
        self,
        vehicle_id: str,
        time_step: float,
        actions: dict[str, dict[str, Any]],
    ) -> None:
        vehicle = self.vehicle_manager.get_vehicle(vehicle_id)
        plan = self._active_plans[vehicle_id]
        result = self._calculate_movement(vehicle, plan, time_step)

        if result.remove_plan:
            self._active_plans.pop(vehicle_id, None)

        actions[vehicle_id] = {
            "action": "move",
            "params": {
                "distance_km": result.distance_km,
                "travel_time": result.travel_time,
                "current_node_id": result.current_node_id,
                "next_node_id": result.next_node_id,
                "edge_progress_km": result.edge_progress_km,
                "status": result.status,
            },
        }

    def _calculate_movement(
        self,
        vehicle: Vehicle,
        plan: MobilityPlan,
        time_step: float,
    ) -> MovementResult:
        remaining_time = max(float(time_step), 0.0)
        remaining_energy_distance = vehicle.available_distance_km()
        result = MovementResult()
        reached_node_id: Optional[int] = None

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

            reached_node_id = next_node
            plan.next_node_index += 1
            plan.current_edge_remaining_km = None

        # 不管这一步是走完了一整条边、只走了半条边、还是完全没动，都从plan当前
        # 的状态（next_node_index/current_edge_remaining_km）把"现在具体在哪"
        # 算出来，每次都汇报current_node_id/next_node_id/edge_progress_km，
        # 不再是"只有到达完整节点才汇报位置"（这是原来代码的一个已知缺口）。
        if plan.next_node_index < len(plan.path):
            position_current_node = plan.path[plan.next_node_index - 1]
            position_next_node = plan.path[plan.next_node_index]
            edge = self.road_network.get_edge_between(position_current_node, position_next_node)
            remaining = plan.current_edge_remaining_km
            edge_progress_km = 0.0 if remaining is None else edge["length_km"] - remaining

            result.current_node_id = position_current_node
            result.next_node_id = position_next_node
            result.edge_progress_km = edge_progress_km
        else:
            # 已经到终点：next_node_id故意设成跟current_node_id一样（而不是None），
            # 是因为move()对"传None"的约定是"这个字段不变"，如果这里传None，
            # next_node_id会停留在到达终点前那一刻的旧值，变成一个过期数据。
            # 设成等于current_node_id，直观地表示"不再朝任何地方走"，同时避免
            # 改动已经定稿过的Vehicle.move()的参数语义。
            result.current_node_id = plan.path[-1]
            result.next_node_id = plan.path[-1]
            result.edge_progress_km = 0.0

        if reached_node_id == vehicle.destination_node_id:
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
