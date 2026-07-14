"""车辆组件，负责保存车辆状态并消费车辆事件。"""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class VehicleStatus(str, Enum):
    """车辆状态。"""

    IDLE = "idle"
    DRIVING = "driving"
    SEEKING_CHARGE = "seeking_charge"
    QUEUEING = "queueing"
    CHARGING = "charging"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass(frozen=True)
class VehicleEvent:
    """其他组件发送给车辆的状态事件。"""

    vehicle_id: str
    event_type: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Vehicle:
    """单辆汽车的静态参数和动态状态。"""

    vehicle_id: str
    origin_node_id: str
    destination_node_id: str
    battery_capacity_kwh: float = 60.0
    low_soc_threshold: float = 0.2
    target_soc: float = 0.8
    energy_consumption_kwh_per_km: float = 0.18

    current_node_id: Optional[str] = None
    soc: float = 1.0
    status: VehicleStatus = VehicleStatus.IDLE
    total_distance_km: float = 0.0
    total_energy_used_kwh: float = 0.0
    total_travel_time: float = 0.0
    total_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.current_node_id is None:
            self.current_node_id = self.origin_node_id

        self._initial_state = self._state_values()

    def reset(self) -> None:
        """恢复车辆的初始状态。"""
        for name, value in self._initial_state.items():
            setattr(self, name, deepcopy(value))

    def step(
        self,
        time_step: float,
        current_time: float,
        action: Optional[Any] = None,
        events: Optional[list[Any]] = None,
    ) -> list[Any]:
        """消费当前时间步属于本车的事件并更新自身状态。"""
        for event in events or []:
            if not isinstance(event, VehicleEvent):
                continue
            if event.vehicle_id != self.vehicle_id:
                continue

            if event.event_type == "movement":
                self._apply_movement_result(**event.data)
            elif event.event_type == "charging":
                self._apply_charging_result(**event.data)
            elif event.event_type == "status":
                self.status = VehicleStatus(event.data["status"])
            else:
                raise ValueError(f"未知车辆事件类型: {event.event_type}")

        return []

    def get_state(self) -> dict[str, Any]:
        """返回车辆状态副本。"""
        state = self._state_values()
        state["status"] = self.status.value
        return state

    def available_distance_km(self) -> float:
        """查询当前电量对应的理论可行驶距离。"""
        if self.energy_consumption_kwh_per_km <= 0:
            return float("inf")

        available_energy_kwh = self.soc * self.battery_capacity_kwh
        return available_energy_kwh / self.energy_consumption_kwh_per_km

    def _apply_movement_result(
        self,
        distance_km: float,
        travel_time: float,
        current_node_id: Optional[str] = None,
        status: Optional[VehicleStatus] = None,
    ) -> None:
        """应用移动事件；能耗与 SOC 模型后续补充。"""
        self.total_distance_km += max(float(distance_km), 0.0)
        self.total_travel_time += max(float(travel_time), 0.0)

        if current_node_id is not None:
            self.current_node_id = current_node_id
        if status is not None:
            self.status = VehicleStatus(status)

    def _apply_charging_result(
        self,
        energy_kwh: float,
        cost: float,
        status: Optional[VehicleStatus] = None,
    ) -> None:
        """应用充电事件；充电量与 SOC 模型后续补充。"""
        self.total_cost += max(float(cost), 0.0)

        if status is not None:
            self.status = VehicleStatus(status)

    def _state_values(self) -> dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "origin_node_id": self.origin_node_id,
            "destination_node_id": self.destination_node_id,
            "battery_capacity_kwh": self.battery_capacity_kwh,
            "low_soc_threshold": self.low_soc_threshold,
            "target_soc": self.target_soc,
            "energy_consumption_kwh_per_km": self.energy_consumption_kwh_per_km,
            "current_node_id": self.current_node_id,
            "soc": self.soc,
            "status": self.status,
            "total_distance_km": self.total_distance_km,
            "total_energy_used_kwh": self.total_energy_used_kwh,
            "total_travel_time": self.total_travel_time,
            "total_cost": self.total_cost,
        }


class VehicleManager:
    """车辆集合组件；负责收集事件并驱动车辆 step。"""

    def __init__(self, vehicles: Optional[list[Vehicle]] = None) -> None:
        self._vehicles: dict[str, Vehicle] = {}

        for vehicle in vehicles or []:
            if vehicle.vehicle_id in self._vehicles:
                raise ValueError(f"车辆 ID 已存在: {vehicle.vehicle_id}")

            self._vehicles[vehicle.vehicle_id] = vehicle

    def reset(self) -> None:
        """重置所有车辆。"""
        for vehicle in self._vehicles.values():
            vehicle.reset()

    def step(
        self,
        time_step: float,
        current_time: float,
        action: Optional[Any] = None,
        events: Optional[list[Any]] = None,
    ) -> list[Any]:
        """把 Env 收集的事件交给对应车辆处理。"""
        vehicle_events: dict[str, list[VehicleEvent]] = {}
        for event in events or []:
            if isinstance(event, VehicleEvent):
                vehicle_events.setdefault(event.vehicle_id, []).append(event)

        for vehicle_id, vehicle in self._vehicles.items():
            vehicle.step(
                time_step=time_step,
                current_time=current_time,
                action=None,
                events=vehicle_events.get(vehicle_id, []),
            )

        return []

    def get_state(self) -> dict[str, Any]:
        """返回车辆集合状态。"""
        status_counts: dict[str, int] = {}
        for vehicle in self._vehicles.values():
            status = vehicle.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "vehicle_count": len(self._vehicles),
            "status_counts": status_counts,
            "vehicles": {
                vehicle_id: vehicle.get_state()
                for vehicle_id, vehicle in self._vehicles.items()
            },
        }

    def get_vehicle(self, vehicle_id: str) -> Vehicle:
        """查询一个车辆并返回副本。"""
        return deepcopy(self._vehicles[vehicle_id])
