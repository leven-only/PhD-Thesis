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

    # 私有状态更新函数
    def _update_internal_status(self) -> None:
        """根据车辆自身状态做内部状态检查。"""
        if self.needs_charge() and self.status == VehicleStatus.DRIVING:
            self.status = VehicleStatus.SEEKING_CHARGE

    # 环境生命周期接口
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
        """推进车辆一个时间步。"""
        self._update_internal_status()

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

    # 状态查询
    def needs_charge(self) -> bool:
        """判断车辆是否需要充电。"""
        return self.soc <= self.low_soc_threshold

    def available_distance_km(self) -> float:
        """根据当前电量返回车辆理论可行驶距离。"""
        if self.energy_consumption_kwh_per_km <= 0:
            return float("inf")

        available_energy_kwh = self.soc * self.battery_capacity_kwh
        return available_energy_kwh / self.energy_consumption_kwh_per_km

    # 判断车辆当前是否可以行驶。可以移动返回true，否则返回false。
    def can_move(self) -> bool:
        return self.status not in (
            VehicleStatus.FINISHED,
            VehicleStatus.FAILED,
            VehicleStatus.CHARGING,
            VehicleStatus.QUEUEING,
        )

    def is_driving(self) -> bool:
        """判断车辆是否处于行驶状态。"""
        return self.status == VehicleStatus.DRIVING

    def is_finished(self) -> bool:
        """判断车辆是否已经完成出行。"""
        return self.status == VehicleStatus.FINISHED

    def is_failed(self) -> bool:
        """判断车辆是否已经失败。"""
        return self.status == VehicleStatus.FAILED

    # 状态统一更新
    def update_state(self, **updates: Any) -> None:
        """统一更新车辆状态；当前只做字段存在性检查，后续可补充特殊约束。"""
        for field_name, value in updates.items():
            if not hasattr(self, field_name):
                raise AttributeError(f"车辆不存在状态字段: {field_name}")

            setattr(self, field_name, value)

    def apply_movement_result(
        self,
        distance_km: float,
        travel_time: float,
        current_node_id: Optional[str] = None,
        status: Optional[VehicleStatus] = None,
    ) -> None:
        """根据一次移动结果，统一更新车辆状态。"""
        safe_distance = max(distance_km, 0.0)
        safe_travel_time = max(travel_time, 0.0)
        energy_used = safe_distance * self.energy_consumption_kwh_per_km
        soc_drop = energy_used / self.battery_capacity_kwh
        new_soc = max(0.0, self.soc - soc_drop)
        new_status = status if status is not None else self.status

        if new_soc <= 0.0:
            new_status = VehicleStatus.FAILED

        updates: dict[str, Any] = {
            "soc": new_soc,
            "status": new_status,
            "total_distance_km": self.total_distance_km + safe_distance,
            "total_energy_used_kwh": self.total_energy_used_kwh + energy_used,
            "total_travel_time": self.total_travel_time + safe_travel_time,
        }

        if current_node_id is not None:
            updates["current_node_id"] = current_node_id

        self.update_state(**updates)

    def apply_charging_result(
        self,
        energy_kwh: float,
        cost: float,
        status: Optional[VehicleStatus] = None,
    ) -> None:
        """根据一次充电结果，统一更新车辆状态。"""
        safe_energy = max(energy_kwh, 0.0)
        safe_cost = max(cost, 0.0)
        soc_gain = safe_energy / self.battery_capacity_kwh
        new_soc = min(1.0, self.soc + soc_gain)
        new_status = status if status is not None else self.status

        self.update_state(
            soc=new_soc,
            status=new_status,
            total_cost=self.total_cost + safe_cost,
        )

# 车辆管理器，用于统一管理车辆集合并提供 env 调用接口。
class VehicleManager:
    def __init__(self, vehicles: Optional[list[Vehicle]] = None) -> None:
        self.vehicles: dict[str, Vehicle] = {}  # 汽车对象
        if vehicles:    # 初始化汽车对象
            for vehicle in vehicles:
                self.add_vehicle(vehicle)

    # 添加一辆车。
    def add_vehicle(self, vehicle: Vehicle) -> None:
        if vehicle.vehicle_id in self.vehicles:
            raise ValueError(f"车辆 ID 已存在: {vehicle.vehicle_id}")

        self.vehicles[vehicle.vehicle_id] = vehicle

    # 根据 ID 获取车辆。
    def get_vehicle(self, vehicle_id: str) -> Vehicle:
        return self.vehicles[vehicle_id]

    # 重置所有车辆。
    def reset(self) -> None:
        for vehicle in self.vehicles.values():
            vehicle.reset()

    # 推进所有车辆一个时间步。
    def step(self, time_step: float, current_time: float, action: Optional[Any] = None) -> None:
        for vehicle in self.vehicles.values():
            vehicle.step(time_step, current_time, action)

    # 返回车辆集合状态。
    def get_state(self) -> dict[str, Any]:
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
