"""充电站模块，用于描述站点位置、充电桩容量、排队队列和充电服务过程。"""

from dataclasses import dataclass, field
from typing import Any, Optional

from sim_env.vehicle import VehicleManager, VehicleStatus


@dataclass
class ChargingStation:
    """单个充电站的基础模型。"""

    station_id: str  # 充电站唯一标识
    node_id: str  # 充电站所在路网节点
    charger_count: int  # 可同时服务的充电桩数量
    charging_power_kw: float  # 单个充电桩功率
    price_per_kwh: float  # 电价

    queue_vehicle_ids: list[str] = field(default_factory=list)  # 等待充电车辆
    charging_vehicle_ids: list[str] = field(default_factory=list)  # 正在充电车辆

    def reset(self) -> None:
        """清空站点排队和充电状态。"""
        self.queue_vehicle_ids = []
        self.charging_vehicle_ids = []

    def has_available_charger(self) -> bool:
        """判断站点是否还有空闲充电桩。"""
        return len(self.charging_vehicle_ids) < self.charger_count

    def contains_vehicle(self, vehicle_id: str) -> bool:
        """判断车辆是否已经在本站排队或充电。"""
        return (
            vehicle_id in self.queue_vehicle_ids
            or vehicle_id in self.charging_vehicle_ids
        )

    def request_charge(self, vehicle_id: str) -> None:
        """接收车辆充电请求；有空桩则充电，否则进入队列。"""
        if self.contains_vehicle(vehicle_id):
            return

        if self.has_available_charger():
            self.charging_vehicle_ids.append(vehicle_id)
            return

        self.queue_vehicle_ids.append(vehicle_id)

    def fill_available_chargers(self) -> None:
        """用排队车辆填充空闲充电桩。"""
        while self.queue_vehicle_ids and self.has_available_charger():
            vehicle_id = self.queue_vehicle_ids.pop(0)
            self.charging_vehicle_ids.append(vehicle_id)

    def remove_vehicle(self, vehicle_id: str) -> None:
        """从站点中移除指定车辆。"""
        if vehicle_id in self.queue_vehicle_ids:
            self.queue_vehicle_ids.remove(vehicle_id)

        if vehicle_id in self.charging_vehicle_ids:
            self.charging_vehicle_ids.remove(vehicle_id)

    def get_state(self) -> dict[str, Any]:
        """返回单个站点状态。"""
        return {
            "station_id": self.station_id,
            "node_id": self.node_id,
            "charger_count": self.charger_count,
            "charging_power_kw": self.charging_power_kw,
            "price_per_kwh": self.price_per_kwh,
            "queue_length": len(self.queue_vehicle_ids),
            "charging_count": len(self.charging_vehicle_ids),
            "queue_vehicle_ids": list(self.queue_vehicle_ids),
            "charging_vehicle_ids": list(self.charging_vehicle_ids),
        }


class StationManager:
    """充电站管理器，用于统一管理站点集合并提供 env 调用接口。"""

    def __init__(
        self,
        vehicle_manager: VehicleManager,
        stations: Optional[list[ChargingStation]] = None,
    ) -> None:
        self.vehicle_manager = vehicle_manager
        self.stations: dict[str, ChargingStation] = {}

        for station in stations or []:
            self.add_station(station)

    def _apply_action(self, action: Optional[Any]) -> None:
        """读取外部传入的充电请求。"""
        if not isinstance(action, dict):
            return

        requests = action.get("charging_requests")

        if not isinstance(requests, dict):
            return

        for vehicle_id, station_id in requests.items():
            self.request_charge(vehicle_id, station_id)

    def _charge_station_vehicles(
        self,
        station: ChargingStation,
        time_step: float,
    ) -> None:
        """推进单个站点内所有正在充电的车辆。"""
        energy_per_vehicle = station.charging_power_kw * max(time_step, 0.0) / 3600

        for vehicle_id in list(station.charging_vehicle_ids):
            vehicle = self.vehicle_manager.get_vehicle(vehicle_id)
            cost = energy_per_vehicle * station.price_per_kwh
            next_status = VehicleStatus.CHARGING

            if vehicle.soc + energy_per_vehicle / vehicle.battery_capacity_kwh >= vehicle.target_soc:
                next_status = VehicleStatus.IDLE

            vehicle.apply_charging_result(
                energy_kwh=energy_per_vehicle,
                cost=cost,
                status=next_status,
            )

            if next_status == VehicleStatus.IDLE:
                station.remove_vehicle(vehicle_id)

        station.fill_available_chargers()

    def add_station(self, station: ChargingStation) -> None:
        """添加一个充电站。"""
        if station.station_id in self.stations:
            raise ValueError(f"充电站 ID 已存在: {station.station_id}")

        self.stations[station.station_id] = station

    def get_station(self, station_id: str) -> ChargingStation:
        """根据 ID 获取充电站。"""
        return self.stations[station_id]

    def request_charge(self, vehicle_id: str, station_id: str) -> None:
        """给指定车辆登记充电站请求。"""
        vehicle = self.vehicle_manager.get_vehicle(vehicle_id)
        station = self.get_station(station_id)

        if vehicle.current_node_id != station.node_id:
            raise ValueError(
                f"车辆 {vehicle_id} 不在充电站节点 {station.node_id}，无法充电"
            )

        station.request_charge(vehicle_id)

        if vehicle_id in station.charging_vehicle_ids:
            vehicle.update_state(status=VehicleStatus.CHARGING)
        else:
            vehicle.update_state(status=VehicleStatus.QUEUEING)

    def reset(self) -> None:
        """重置所有充电站。"""
        for station in self.stations.values():
            station.reset()

    def step(self, time_step: float, current_time: float, action: Optional[Any] = None) -> None:
        """推进所有充电站一个时间步。"""
        self._apply_action(action)

        for station in self.stations.values():
            self._charge_station_vehicles(station, time_step)

    def get_state(self) -> dict[str, Any]:
        """返回所有充电站状态。"""
        return {
            "station_count": len(self.stations),
            "stations": {
                station_id: station.get_state()
                for station_id, station in self.stations.items()
            },
        }
