"""充电站组件：站点属性 + 按时间刷新的动态运行状态。

站点信息是抽象、整体层面的——不追踪具体哪个充电桩被哪辆车占用，只用排队论
里的话务强度（offered load）ρ = λ(t)×μ 表示"平均意义上这个 charger type
有多拥堵"，用它推出期望空闲桩数、拥堵服务费、期望等待时间。ρ 越大代表
越拥堵，空闲桩越少、等待时间越长，这跟"真实精确计数"是两种不同粒度的
建模，这里选择后者的抽象版本。

注意 proposal 公式 (3.5) 里的 ρ 就是这个"期望忙碌充电桩数"（offered load，
量级跟 num_chargers 一样大，不是 0~1 之间的比例），跟一般 M/M/k 记号里
"利用率 = λ/(kμ) ∈[0,1)"是两个概念——本文件内部区分为：
    _utilization()  ->  标准化利用率 ρ_norm = λ/(kμ) ∈[0,1)，仅用于内部推导
                        offered_load，不直接对外暴露；
    offered_load    ->  ρ_norm × num_chargers = λ×μ，proposal 公式 (3.5) 里
                        真正参与 available_chargers 和 waiting_time 计算的量。
两者数值上满足 offered_load = num_chargers × _utilization()，是同一件事的
两种刻度，不是两套独立参数。

因此 ChargingStation 现在完全是"世界状态"角色，不再有任何按事件触发的更新：
reset() / step(current_time, tou_tariff) 就是全部对外接口，`available_chargers`
/`dynamic_service_fee`/`waiting_time_minutes` 都是每次调用时用 current_time
对应的电价和静态参数（λ、μ、σ、k）现算出来的派生量。

如果 arrival_rate_per_hour 以后要做成按时间变化的表（类似 TOU tariff 的
高低峰），ρ 以及后面几个派生量就会跟着自然随时间波动；现在先假定它是每个
charger type 的一个固定常数，所以这几个派生量在整个仿真过程中不会变化，只有
total_price 会因为 tou_tariff 随时间变化而变化。

以下三个派生量已按 proposal 3.4.2 节原文公式实现，均已核对截图确认：
    dynamic_service_fee  <- 公式 (3.3)：f = β(c-a)/c （H. Wu et al., 2025）
    total_price           <- 公式 (3.4)：p = p_TOU + f
    waiting_time_minutes  <- 公式 (3.5)：Allen-Cunneen 近似 M/G/k 等待时间
                              （Zhong et al., 2023）
"""

import math
from copy import deepcopy
from typing import Any, Optional


class ChargingStation:
    """单个充电站；站点级别的抽象状态，不追踪具体充电桩占用。"""

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

        # 充电桩的静态配置，按 charger_type 拆成几个字典属性。
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
        self.available_chargers: dict[str, float] = {ct: 0.0 for ct in self.num_chargers}
        self.dynamic_service_fee: dict[str, float] = {ct: 0.0 for ct in self.num_chargers}
        self.total_price: dict[str, float] = {ct: 0.0 for ct in self.num_chargers}
        self.waiting_time_minutes: dict[str, float] = {ct: 0.0 for ct in self.num_chargers}

        self._recompute_derived_state()

    # ------------------------------------------------------------------
    # 公共接口：reset / step / get_state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """电价清零，重新计算全部派生量。"""
        self.tou_tariff = 0.0
        self._recompute_derived_state()

    def step(self, current_time: float, tou_tariff: float = 0.0) -> None:
        """按当前时间刷新分时电价，并重新计算全部派生量。

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
    # 内部实现（下划线开头，外部代码不应依赖）
    # ------------------------------------------------------------------

    def _recompute_derived_state(self) -> None:
        for charger_type in self.num_chargers:
            utilization = self._utilization(charger_type)
            offered_load = utilization * self.num_chargers[charger_type]

            available = self.num_chargers[charger_type] - offered_load
            self.available_chargers[charger_type] = available

            fee = self._estimate_dynamic_service_fee(charger_type, available)
            self.dynamic_service_fee[charger_type] = fee
            self.total_price[charger_type] = self.tou_tariff + fee

            self.waiting_time_minutes[charger_type] = self._estimate_waiting_time_minutes(
                charger_type, offered_load
            )

    def _utilization(self, charger_type: str) -> float:
        """标准化利用率 ρ_norm = λ/(kμ) ∈[0,1)，仅用于内部推导 offered_load。"""
        arrival_rate_per_hour = self.arrival_rate_per_hour[charger_type]
        mean_service_time_hours = self.mean_service_time_minutes[charger_type] / 60
        num_chargers = self.num_chargers[charger_type]
        return arrival_rate_per_hour * mean_service_time_hours / num_chargers

    def _estimate_dynamic_service_fee(self, charger_type: str, available_chargers: float) -> float:
        """proposal 公式 (3.3)：拥堵敏感服务费 f = β(c-a)/c（H. Wu et al., 2025）。

        c = num_chargers（静态充电桩数），a = available_chargers（当前期望
        空闲桩数）。可用桩越少（c-a 越大），服务费越高；β 即
        marginal_cost_factor，默认 0.2。
        """
        num_chargers = self.num_chargers[charger_type]
        return self.marginal_cost_factor * (num_chargers - available_chargers) / num_chargers

    def _estimate_waiting_time_minutes(self, charger_type: str, offered_load: float) -> float:
        """proposal 公式 (3.5)：Allen-Cunneen 近似的 M/G/k 等待时间，单位：分钟。

        T_wait = (σ²+μ²) / (2μ(c-ρ)) × [1 + Σ_{h=0}^{c-1} (c-1)!(c-ρ) / (h!·ρ^(c-h-1))]^(-1)

        其中 c = num_chargers，ρ = offered_load（即 λ×μ，未除以 c），
        μ = mean_service_time_minutes，σ = service_time_std_minutes。
        方括号里的求和项是 Erlang-C 公式的分母，取倒数后就是"需要排队等待"
        的概率；乘以前面的 (σ²+μ²)/(2μ(c-ρ)) 就是 Allen-Cunneen 对一般服务
        时间分布（而不是纯指数分布）的修正。
        """
        num_chargers = self.num_chargers[charger_type]
        mean_service_time = self.mean_service_time_minutes[charger_type]
        service_time_std = self.service_time_std_minutes[charger_type]

        if offered_load <= 0 or mean_service_time <= 0 or num_chargers <= 0:
            return 0.0
        if offered_load >= num_chargers:
            # 违反稳定性条件 c > ρ（到达率超过服务能力），保守截断避免除零/发散。
            offered_load = num_chargers - 1e-6

        erlang_c_denominator = 1.0
        for h in range(num_chargers):
            erlang_c_denominator += (
                math.factorial(num_chargers - 1)
                * (num_chargers - offered_load)
                / (math.factorial(h) * offered_load ** (num_chargers - h - 1))
            )
        probability_of_wait = 1.0 / erlang_c_denominator

        prefactor = (service_time_std**2 + mean_service_time**2) / (
            2 * mean_service_time * (num_chargers - offered_load)
        )
        return prefactor * probability_of_wait


class StationManager:
    """站点集合组件；负责把分时电价分发给各站点。"""

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
