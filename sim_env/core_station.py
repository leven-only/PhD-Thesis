"""充电站环境，负责保存站点状态并响应环境更新事件。"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Optional, Union


@dataclass
class ChargerTypeConfig:
    """一种充电桩的固定配置。"""

    charger_type: str
    num_chargers: int
    power_kw: float


class ChargingStation:
    """单个充电站；属性只能在 step 中更新。"""

    def __init__(
        self,
        station_id: str,
        mapped_node: str,
        chargers: dict[str, Union[ChargerTypeConfig, dict[str, Any]]],
        access_distance_km: float = 0.0,
        **attributes: Any,
    ) -> None:
        self.station_id = station_id
        self.mapped_node = mapped_node
        self.access_distance_km = float(access_distance_km)
        self.chargers = {
            charger_type: (
                deepcopy(config)
                if isinstance(config, ChargerTypeConfig)
                else ChargerTypeConfig(
                    charger_type=charger_type,
                    num_chargers=int(config["num_chargers"]),
                    power_kw=float(config.get("power_kw", config.get("power"))),
                )
            )
            for charger_type, config in chargers.items()
        }

        if not self.chargers:
            raise ValueError("充电站至少需要配置一种 charger type")

        defaults: dict[str, Union[int, float]] = {
            "waiting_time_seconds": 0.0,
            "arrival_rate": 0.0,
            "service_time_seconds": 0.0,
            "tou_tariff": 0.0,
            "dynamic_service_fee": 0.0,
            "occupied_chargers": 0,
        }
        unknown = set(attributes) - set(defaults)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"未知充电站属性: {names}")

        for name, default in defaults.items():
            supplied = attributes.get(name, {})
            if not isinstance(supplied, dict):
                raise TypeError(f"属性 {name} 必须按 charger type 保存")

            setattr(
                self,
                name,
                {
                    charger_type: deepcopy(supplied.get(charger_type, default))
                    for charger_type in self.chargers
                },
            )

        self._initial_state = self._public_state()

    def reset(self) -> None:
        """恢复构造完成时的站点状态。"""
        for name, value in self._initial_state.items():
            setattr(self, name, deepcopy(value))

    def step(
        self,
        time_step: float,
        current_time: float,
        action: Optional[Any] = None,
    ) -> None:
        """应用当前时间步属于本站的属性更新事件。"""
        if not isinstance(action, dict):
            return

        for charger_type, updates in action.items():
            if charger_type not in self.chargers:
                raise ValueError(
                    f"充电站 {self.station_id} 不支持 charger type: {charger_type}"
                )
            if not isinstance(updates, dict):
                raise TypeError("充电站属性更新必须是字典")

            for name, value in updates.items():
                if name.startswith("_") or not hasattr(self, name):
                    raise AttributeError(f"未知充电站属性: {name}")
                if name == "chargers":
                    raise AttributeError("不允许在运行期间修改充电桩配置")

                current_value = getattr(self, name)
                if not isinstance(current_value, dict):
                    raise AttributeError(f"不允许更新固定属性: {name}")

                current_value[charger_type] = deepcopy(value)

    def get_state(self) -> dict[str, Any]:
        """返回站点状态副本。"""
        return {
            **self._public_state(),
            "chargers": {
                charger_type: asdict(config)
                for charger_type, config in self.chargers.items()
            },
        }

    def _public_state(self) -> dict[str, Any]:
        return {
            name: deepcopy(value)
            for name, value in vars(self).items()
            if not name.startswith("_") and name != "chargers"
        }


class StationManager:
    """站点集合组件；负责把环境事件分发给各站点。"""

    def __init__(
        self,
        stations: Optional[list[ChargingStation]] = None,
    ) -> None:
        self._stations: dict[str, ChargingStation] = {}

        for station in stations or []:
            if station.station_id in self._stations:
                raise ValueError(f"充电站 ID 已存在: {station.station_id}")

            self._stations[station.station_id] = station

    def reset(self) -> None:
        """重置所有站点。"""
        for station in self._stations.values():
            station.reset()

    def step(
        self,
        time_step: float,
        current_time: float,
        action: Optional[Any] = None,
    ) -> None:
        """分发当前时间步的站点更新事件。"""
        if not isinstance(action, dict):
            return

        station_updates = action.get("station_updates")
        if not isinstance(station_updates, dict):
            return

        for station_id, updates in station_updates.items():
            if station_id not in self._stations:
                raise ValueError(f"充电站不存在: {station_id}")

            self._stations[station_id].step(
                time_step=time_step,
                current_time=current_time,
                action=updates,
            )

    def get_state(self) -> dict[str, Any]:
        """返回全部站点状态。"""
        return {
            "station_count": len(self._stations),
            "stations": {
                station_id: station.get_state()
                for station_id, station in self._stations.items()
            },
        }

    def get_station(self, station_id: str) -> ChargingStation:
        """查询一个站点并返回副本。"""
        return deepcopy(self._stations[station_id])
