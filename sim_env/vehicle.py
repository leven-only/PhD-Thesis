"""车辆模块，用于描述电动汽车状态、路径移动、能耗计算、充电需求和行为状态机。"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class VehicleStatus(str, Enum):
    """车辆在仿真中的基础状态。"""

    IDLE = "idle"
    DRIVING = "driving"
    SEEKING_CHARGE = "seeking_charge"
    QUEUEING = "queueing"
    CHARGING = "charging"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass
class Vehicle:
    """汽车静态信息"""

    vehicle_id: str  # 汽车唯一标识
    origin_node_id: str  # 起点
    destination_node_id: str  # 终点
    battery_capacity_kwh: float = 60.0  # 电池总容量
    low_soc_threshold: float = 0.2  # 最低SOC
    target_soc: float = 0.8  # 目标SOC
    energy_consumption_kwh_per_km: float = 0.18  # 能耗率，后期替换为真实的能耗模型

    """汽车实时信息"""

    current_node_id: Optional[str] = None  # 当前位置
    soc: float = 1.0  # 当前SOC
    status: VehicleStatus = VehicleStatus.IDLE  # 当前状态
    total_distance_km: float = 0.0  # 累计行驶距离
    total_energy_used_kwh: float = 0.0  # 累计能耗
    total_travel_time: float = 0.0  # 累计行驶时间
    total_cost: float = 0.0  # 累计费用

    def __post_init__(self) -> None:
        if self.current_node_id is None:
            self.current_node_id = self.origin_node_id

        self.initial_soc = self.soc
        self.initial_status = self.status
        self.initial_node_id = self.current_node_id
        self.initial_total_cost = self.total_cost

    def reset(self) -> None:
        """将车辆恢复到初始状态。"""
        self.current_node_id = self.initial_node_id
        self.soc = self.initial_soc
        self.status = self.initial_status
        self.total_distance_km = 0.0
        self.total_energy_used_kwh = 0.0
        self.total_travel_time = 0.0
        self.total_cost = self.initial_total_cost

    def step(self, time_step: float, current_time: float, action: Optional[Any] = None) -> None:
        """推进车辆一个时间步。

        车辆模型只维护车辆自身状态，不在这里选择路线或充电站。
        路线规划和站点选择应由算法或环境执行层完成。
        """
        if self.needs_charge() and self.status == VehicleStatus.DRIVING:
            self.status = VehicleStatus.SEEKING_CHARGE

    def consume_energy(self, distance_km: float) -> float:
        """根据行驶距离扣减电量，并返回本次消耗的电量。"""
        energy_used = distance_km * self.energy_consumption_kwh_per_km
        soc_drop = energy_used / self.battery_capacity_kwh

        self.soc = max(0.0, self.soc - soc_drop)
        self.total_distance_km += distance_km
        self.total_energy_used_kwh += energy_used

        if self.soc <= 0.0:
            self.status = VehicleStatus.FAILED

        return energy_used

    def record_travel(self, distance_km: float, travel_time: float) -> float:
        """记录一次行驶过程，并同步扣减对应能耗。"""
        self.total_travel_time += max(travel_time, 0.0)
        return self.consume_energy(distance_km)

    def move_to_node(self, node_id: str) -> None:
        """更新车辆当前位置。"""
        self.current_node_id = node_id

    def charge(self, energy_kwh: float) -> float:
        """给车辆充电，并返回实际充入的电量。"""
        available_capacity = (1.0 - self.soc) * self.battery_capacity_kwh
        charged_energy = min(max(energy_kwh, 0.0), available_capacity)

        self.soc += charged_energy / self.battery_capacity_kwh

        if self.soc >= self.target_soc and self.status == VehicleStatus.CHARGING:
            self.status = VehicleStatus.DRIVING

        return charged_energy

    def needs_charge(self) -> bool:
        """判断车辆是否需要充电。"""
        return self.soc <= self.low_soc_threshold

    def add_cost(self, cost: float) -> None:
        """增加车辆累计费用。"""
        self.total_cost += max(cost, 0.0)

    def set_status(self, status: VehicleStatus) -> None:
        """更新车辆当前状态。"""
        self.status = status

    def mark_queueing(self) -> None:
        """标记车辆正在充电站排队。"""
        self.status = VehicleStatus.QUEUEING

    def mark_charging(self) -> None:
        """标记车辆正在充电。"""
        self.status = VehicleStatus.CHARGING

    def mark_finished(self) -> None:
        """标记车辆已经完成出行。"""
        self.status = VehicleStatus.FINISHED

    def get_state(self) -> dict[str, Any]:
        """返回车辆当前状态。"""
        return {
            "vehicle_id": self.vehicle_id,
            "origin_node_id": self.origin_node_id,
            "destination_node_id": self.destination_node_id,
            "current_node_id": self.current_node_id,
            "soc": self.soc,
            "status": self.status.value,
            "total_distance_km": self.total_distance_km,
            "total_energy_used_kwh": self.total_energy_used_kwh,
            "total_travel_time": self.total_travel_time,
            "total_cost": self.total_cost,
        }


class VehicleManager:
    """车辆管理器，用于统一管理车辆集合并提供 env 调用接口。"""

    def __init__(self, vehicles: Optional[list[Vehicle]] = None) -> None:
        self.vehicles: dict[str, Vehicle] = {}

        if vehicles:
            for vehicle in vehicles:
                self.add_vehicle(vehicle)

    def add_vehicle(self, vehicle: Vehicle) -> None:
        """添加一辆车。"""
        if vehicle.vehicle_id in self.vehicles:
            raise ValueError(f"车辆 ID 已存在: {vehicle.vehicle_id}")

        self.vehicles[vehicle.vehicle_id] = vehicle

    def get_vehicle(self, vehicle_id: str) -> Vehicle:
        """根据 ID 获取车辆。"""
        return self.vehicles[vehicle_id]

    def reset(self) -> None:
        """重置所有车辆。"""
        for vehicle in self.vehicles.values():
            vehicle.reset()

    def step(self, time_step: float, current_time: float, action: Optional[Any] = None) -> None:
        """推进所有车辆一个时间步。"""
        for vehicle in self.vehicles.values():
            vehicle.step(time_step, current_time, action)

    def get_state(self) -> dict[str, Any]:
        """返回车辆集合状态。"""
        status_counts: dict[str, int] = {}

        for vehicle in self.vehicles.values():
            status = vehicle.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "vehicle_count": len(self.vehicles),
            "status_counts": status_counts,
            "vehicles": {
                vehicle_id: vehicle.get_state()
                for vehicle_id, vehicle in self.vehicles.items()
            },
        }
