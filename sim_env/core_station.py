"""充电站组件：站点属性 + 按时间刷新的动态运行状态。

充电桩信息拍平成站点自己的属性（num_chargers、power_kw 等，都是按
charger_type 分的字典），不单独用一个类包装。

角色说明：ChargingStation 的更新分两种，触发时机不一样——
分时电价(TOU tariff)、服务费、总价、等待时间这些是按时间刷新的派生量，
由 step(current_time, tou_tariff) 统一重算，属于定时触发；
空闲充电桩数量(available_chargers)是离散事件（某辆车开始/结束充电）
发生的那一刻就要立刻变化的，由 occupy()/release() 直接修改，不等下一次
step()。这两种触发时机不一样，所以没有合并成一个接口。

occupy()/release() 现在只做加减，不做越界检查、也不触发派生量重算——
数据由使用方自己保证正确，派生量统一在 step() 里按最新的占用情况重算。

dynamic_service_fee 和 waiting_time_minutes 的具体公式对应 proposal 3.4.2
节的拥堵敏感定价和 M/G/k 排队模型，这里先留出占位实现（返回 0），实现的
时候再把公式抄进去。
"""

from copy import deepcopy
from typing import Any, Optional


class ChargingStation:
    """单个充电站；只保存状态、按外部信号更新，不做任何充电分配决策。"""

    def __init__(
        self,
        station_id: str,
        mapped_node: int,
        chargers: dict[str, dict[str, Any]],
        access_distance_km: float = 0.0,
        marginal_cost_factor: float = 0.2,
    ) -> None:
        self.station_id = station_id
        self.mapped_node = mapped_node
        self.access_distance_km = access_distance_km
        self.marginal_cost_factor = marginal_cost_factor

        # 充电桩的静态配置，按 charger_type 拆成几个字典属性，不再包一层类。
        self.num_chargers = {ct: cfg["num_chargers"] for ct, cfg in chargers.items()}
        self.power_kw = {ct: cfg["power_kw"] for ct, cfg in chargers.items()}
        self.arrival_rate_per_hour = {
            ct: cfg.get("arrival_rate_per_hour", 0.0) for ct, cfg in chargers.items()
        }
        self.mean_service_time_minutes = {
            ct: cfg.get("mean_service_time_minutes", 0.0) for ct, cfg in chargers.items()
        }
        self.service_time_std_minutes = {
            ct: cfg.get("service_time_std_minutes", 0.0) for ct, cfg in chargers.items()
        }

        self.tou_tariff = 0.0
        self.available_chargers: dict[str, int] = dict(self.num_chargers)
        self.dynamic_service_fee: dict[str, float] = {ct: 0.0 for ct in self.num_chargers}
        self.total_price: dict[str, float] = {ct: 0.0 for ct in self.num_chargers}
        self.waiting_time_minutes: dict[str, float] = {ct: 0.0 for ct in self.num_chargers}

        self._initial_available_chargers = dict(self.available_chargers)

    # ------------------------------------------------------------------
    # 按时间触发：reset / step / get_state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """把占用数恢复到初始值，并按 tou_tariff=0 重新计算派生量。"""
        self.available_chargers = dict(self._initial_available_chargers)
        self.tou_tariff = 0.0
        self._recompute_derived_state()

    def step(self, current_time: float, tou_tariff: float = 0.0) -> None:
        """按当前时间刷新分时电价，并重新计算服务费、总价、等待时间。

        current_time 目前只用来标记这是哪个 time slot；tou_tariff 是分时
        电价表在这个时间点查出来的值，由调用方（StationManager）传入，
        因为电价是所有站点共享的一张表，不需要每个站点各自查一遍。
        """
        self.tou_tariff = tou_tariff
        self._recompute_derived_state()

    def get_state(self) -> dict[str, Any]:
        """返回站点状态副本。"""
        return {
            "station_id": self.station_id,
            "mapped_node": self.mapped_node,
            "access_distance_km": self.access_distance_km,
            "marginal_cost_factor": self.marginal_cost_factor,
            "num_chargers": dict(self.num_chargers),
            "power_kw": dict(self.power_kw),
            "arrival_rate_per_hour": dict(self.arrival_rate_per_hour),
            "mean_service_time_minutes": dict(self.mean_service_time_minutes),
            "service_time_std_minutes": dict(self.service_time_std_minutes),
            "tou_tariff": self.tou_tariff,
            "available_chargers": dict(self.available_chargers),
            "dynamic_service_fee": dict(self.dynamic_service_fee),
            "total_price": dict(self.total_price),
            "waiting_time_minutes": dict(self.waiting_time_minutes),
        }

    # ------------------------------------------------------------------
    # 按事件触发：occupy / release
    # ------------------------------------------------------------------

    def occupy(self, charger_type: str) -> None:
        """占用一个充电桩（空闲数减一），立刻生效，不等 step()。"""
        self.available_chargers[charger_type] -= 1

    def release(self, charger_type: str) -> None:
        """释放一个充电桩（空闲数加一），立刻生效，不等 step()。"""
        self.available_chargers[charger_type] += 1

    # ------------------------------------------------------------------
    # 内部实现（下划线开头，外部代码不应依赖）
    # ------------------------------------------------------------------

    def _recompute_derived_state(self) -> None:
        for charger_type in self.num_chargers:
            fee = self._estimate_dynamic_service_fee(charger_type)
            self.dynamic_service_fee[charger_type] = fee
            self.total_price[charger_type] = self.tou_tariff + fee
            self.waiting_time_minutes[charger_type] = self._estimate_waiting_time_minutes(charger_type)

    def _estimate_dynamic_service_fee(self, charger_type: str) -> float:
        """拥堵敏感服务费，对应 proposal 3.4.2 节公式。

        TODO: 目前占位为 0，实现的时候按 marginal_cost_factor 和当前占用率
        （1 - available_chargers / num_chargers）把公式抄进来。
        """
        return 0.0

    def _estimate_waiting_time_minutes(self, charger_type: str) -> float:
        """M/G/k 排队模型估计的等待时间，对应 proposal 3.4.2 节公式。

        TODO: 目前占位为 0，实现的时候用 arrival_rate_per_hour、
        mean_service_time_minutes、service_time_std_minutes、num_chargers
        把公式抄进来。
        """
        return 0.0


class StationManager:
    """站点集合组件；负责把分时电价分发给各站点。

    occupy()/release() 不在这里转发——占用/释放要立刻生效，调用方（以后的
    ChargingManager）需要直接持有具体 ChargingStation 的真实对象来调用，
    而不是通过这里的 get_station()（返回的是深拷贝，改了不影响真实状态）。
    """

    def __init__(self, stations: Optional[list[ChargingStation]] = None) -> None:
        self._stations: dict[str, ChargingStation] = {}

        for station in stations or []:
            if station.station_id in self._stations:
                raise ValueError(f"充电站 ID 已存在: {station.station_id}")

            self._stations[station.station_id] = station

    # ------------------------------------------------------------------
    # 公共接口：reset / step / get_state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """重置所有站点。"""
        for station in self._stations.values():
            station.reset()

    def step(self, current_time: float, tou_tariff: float = 0.0) -> None:
        """把当前时间和分时电价转发给每个站点，让它们各自刷新派生量。"""
        for station in self._stations.values():
            station.step(current_time=current_time, tou_tariff=tou_tariff)

    def get_state(self) -> dict[str, Any]:
        """返回全部站点状态。"""
        return {
            "station_count": len(self._stations),
            "stations": {
                station_id: station.get_state()
                for station_id, station in self._stations.items()
            },
        }

    # ------------------------------------------------------------------
    # 个性化查询接口
    # ------------------------------------------------------------------

    def get_station(self, station_id: str) -> ChargingStation:
        """查询一个站点并返回副本。"""
        if station_id not in self._stations:
            raise ValueError(f"充电站不存在: {station_id}")
        return deepcopy(self._stations[station_id])
