import math
from copy import deepcopy
from typing import Any, Optional


class ChargingStation:
    """单个充电站；站点级别的抽象状态，不追踪具体充电桩占用。"""

    def __init__(
        self,
        num_chargers: list[float],                   # 充电桩数量：[slow, fast]，来自站点物理数据文件
        power_kw: list[float],                        # 充电功率(kW)：[slow, fast]，来自站点物理数据文件
        arrival_rate_per_hour: list[list[float]],     # 到达率时刻表：[slow数组, fast数组]，每个数组长度24，第h位是h点的到达率(辆/小时)，直接来自场景配置文件，不再另外计算
        initial_time: float,                          # 仿真起始时间，由外部传入；reset()会把到达率恢复到这个时间点对应的值，而不是固定回到0点
        station_id: str,                              # 站点id
        mapped_node: int,                             # 映射的路网节点
        access_distance_km: float,                    # 到映射节点的距离(km)
        mean_service_time_minutes: float,             # 平均服务时间(分钟)，站点级别，不分类型
        service_time_std_minutes: float,              # 服务时间标准差(分钟)，站点级别，不分类型
        marginal_cost_factor: float = 0.2,            # 边际成本因子β，全局共享常数，默认0.2（proposal 3.4.2经验取值）
    ) -> None:
        # ---- 与充电桩类型无关的参数 ----
        self.station_id = station_id                      # 站点id
        self.mapped_node = mapped_node                    # 映射的路网节点
        self.access_distance_km = access_distance_km      # 到映射节点的距离(km)
        self.marginal_cost_factor = marginal_cost_factor  # 边际成本因子β，全局共享，不分站点、不分充电桩类型
        self.mean_service_time_minutes = mean_service_time_minutes  # 平均服务时间(分钟)，站点级别，不分充电桩类型
        self.service_time_std_minutes = service_time_std_minutes    # 服务时间标准差(分钟)，站点级别，不分充电桩类型
        self.tou_tariff = 0.0                              # 分时电价，环境信息，来自外部电价表，由StationManager逐time slot注入，构造时先占位
        self._initial_time = initial_time                  # 仿真起始时间，只在构造和reset()时使用，构造后不应再被外部修改

        # ---- slow类型充电桩参数 ----
        self.num_chargers_slow = num_chargers[0]                            # slow充电桩数量
        self.power_kw_slow = power_kw[0]                                    # slow充电功率(kW)
        self._arrival_rate_timetable_slow = arrival_rate_per_hour[0]        # slow到达率时刻表，长度24，来自场景配置文件，不随仿真变化
        self.arrival_rate_per_hour_slow = self._arrival_rate_timetable_slow[self._hour_index(initial_time)]  # slow当前生效到达率，构造时取initial_time对应小时的值，由step()按current_time更新

        self.available_chargers_slow = 0.0                                  # slow当前空闲桩数，动态量，由_recompute_derived_state()更新
        self.dynamic_service_fee_slow = 0.0                                 # slow动态服务费，动态量，由_recompute_derived_state()更新
        self.total_price_slow = 0.0                                         # slow总价，动态量，由_estimate_dynamic_service_fee()更新
        self.waiting_time_minutes_slow = 0.0                                # slow预计等待时间(分钟)，动态量，由_estimate_waiting_time_minutes()更新

        # ---- fast类型充电桩参数 ----
        self.num_chargers_fast = num_chargers[1]                            # fast充电桩数量
        self.power_kw_fast = power_kw[1]                                    # fast充电功率(kW)
        self._arrival_rate_timetable_fast = arrival_rate_per_hour[1]        # fast到达率时刻表，长度24，来自场景配置文件，不随仿真变化
        self.arrival_rate_per_hour_fast = self._arrival_rate_timetable_fast[self._hour_index(initial_time)]  # fast当前生效到达率，构造时取initial_time对应小时的值，由step()按current_time更新

        self.available_chargers_fast = 0.0                                  # fast当前空闲桩数，动态量，由_recompute_derived_state()更新
        self.dynamic_service_fee_fast = 0.0                                 # fast动态服务费，动态量，由_recompute_derived_state()更新
        self.total_price_fast = 0.0                                         # fast总价，动态量，由_estimate_dynamic_service_fee()更新
        self.waiting_time_minutes_fast = 0.0                                # fast预计等待时间(分钟)，动态量，由_estimate_waiting_time_minutes()更新

    # ------------------------------------------------------------------
    # 公共接口：reset / step / get_state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """回到_initial_time这个时刻，并直接调用一次step()完成初始化，因此reset()之后
        全部动态量都是按_initial_time、tou_tariff=0算出来的真实值，不是占位的0.0。
        """
        self.step(current_time=self._initial_time, tou_tariff=0.0)

    def step(self, current_time: float, tou_tariff: float = 0.0) -> None:
        """按当前分时电价、当前小时对应的到达率刷新，重新计算全部动态量。

        current_time按整点取小时、对24取模得到0-23的下标，直接从构造时存好的时刻表里
        取值，不再做任何计算（时刻表里的数就是最终到达率本身）。
        """
        self.tou_tariff = tou_tariff
        hour_index = self._hour_index(current_time)
        self.arrival_rate_per_hour_slow = self._arrival_rate_timetable_slow[hour_index]
        self.arrival_rate_per_hour_fast = self._arrival_rate_timetable_fast[hour_index]
        self._recompute_derived_state()

    def get_state(self) -> dict[str, Any]:
        """返回站点的高层静态信息（站点身份、位置、物理配置），不含随时间变化的动态量；
        动态量（价格、等待时间等）请用下面按充电桩类型查询的几个个性化函数。
        """
        return {
            "station_id": self.station_id,
            "mapped_node": self.mapped_node,
            "access_distance_km": self.access_distance_km,
            "num_chargers_slow": self.num_chargers_slow,
            "power_kw_slow": self.power_kw_slow,
            "num_chargers_fast": self.num_chargers_fast,
            "power_kw_fast": self.power_kw_fast,
        }

    # ------------------------------------------------------------------
    # 个性化查询函数
    # ------------------------------------------------------------------

    def get_charger_type_state(self, charger_type: str) -> dict[str, Any]:
        """按充电桩类型（"slow"/"fast"）查询该类型当前的total_price和waiting_time_minutes。"""
        self._validate_charger_type(charger_type)
        return {
            "total_price": getattr(self, f"total_price_{charger_type}"),
            "waiting_time_minutes": getattr(self, f"waiting_time_minutes_{charger_type}"),
        }

    def get_total_price(self, charger_type: str) -> float:
        """按充电桩类型（"slow"/"fast"）单独查询当前total_price。"""
        self._validate_charger_type(charger_type)
        return getattr(self, f"total_price_{charger_type}")

    def get_waiting_time_minutes(self, charger_type: str) -> float:
        """按充电桩类型（"slow"/"fast"）单独查询当前waiting_time_minutes。"""
        self._validate_charger_type(charger_type)
        return getattr(self, f"waiting_time_minutes_{charger_type}")

    # ------------------------------------------------------------------
    # 内部实现（下划线开头，外部代码不应依赖）
    # ------------------------------------------------------------------

    @staticmethod
    def _hour_index(current_time: float) -> int:
        """把任意时刻（允许跨天累计，如第2天9点=33.0）换算成0-23的小时下标，用于查到达率时刻表。
        构造函数、reset()、step()三处都用这一个函数，避免同样的取整/取模逻辑写三份。
        """
        return int(current_time) % 24

    @staticmethod
    def _validate_charger_type(charger_type: str) -> None:
        """三个个性化查询函数共用的入参校验，charger_type只能是"slow"或"fast"，否则直接报错。"""
        if charger_type not in ("slow", "fast"):
            raise ValueError(f"未知的充电桩类型: {charger_type}，只能是\"slow\"或\"fast\"")

    def _recompute_derived_state(self) -> None:
        """整体入口：依次为slow、fast两种充电桩类型更新全部动态量，注意更新顺序（available -> fee -> price -> waiting）。"""
        for charger_type in ("slow", "fast"):
            self._update_available_chargers(charger_type)
            self._update_dynamic_service_fee(charger_type)
            self._update_total_price(charger_type)
            self._update_waiting_time_minutes(charger_type)

    def _utilization(self, charger_type: str) -> float:
        """标准化利用率 ρ_norm = λ/(kμ) ∈[0,1)，仅供内部推导offered_load使用。"""
        arrival_rate_per_hour = getattr(self, f"arrival_rate_per_hour_{charger_type}")
        mean_service_time_hours = self.mean_service_time_minutes / 60  # 分钟转小时，与λ的"每小时"单位对齐
        num_chargers = getattr(self, f"num_chargers_{charger_type}")
        return arrival_rate_per_hour * mean_service_time_hours / num_chargers

    def _offered_load(self, charger_type: str) -> float:
        """话务强度 ρ = λμ（offered load），量纲与充电桩数相同，供available/waiting两处复用。"""
        num_chargers = getattr(self, f"num_chargers_{charger_type}")
        return self._utilization(charger_type) * num_chargers

    def _update_available_chargers(self, charger_type: str) -> None:
        """Little's Law精确式 a = c - ρ，更新对应类型的available_chargers。"""
        num_chargers = getattr(self, f"num_chargers_{charger_type}")
        offered_load = self._offered_load(charger_type)
        setattr(self, f"available_chargers_{charger_type}", num_chargers - offered_load)

    def _update_dynamic_service_fee(self, charger_type: str) -> None:
        """proposal公式(3.3) f=β(c-a)/c，更新对应类型的dynamic_service_fee；依赖available_chargers已更新。"""
        num_chargers = getattr(self, f"num_chargers_{charger_type}")
        available_chargers = getattr(self, f"available_chargers_{charger_type}")
        fee = self.marginal_cost_factor * (num_chargers - available_chargers) / num_chargers
        setattr(self, f"dynamic_service_fee_{charger_type}", fee)

    def _update_total_price(self, charger_type: str) -> None:
        """proposal公式(3.4) p=tou_tariff+f，更新对应类型的total_price；依赖dynamic_service_fee已更新。"""
        fee = getattr(self, f"dynamic_service_fee_{charger_type}")
        setattr(self, f"total_price_{charger_type}", self.tou_tariff + fee)

    def _update_waiting_time_minutes(self, charger_type: str) -> None:
        """proposal公式(3.5) Allen-Cunneen近似M/G/k等待时间，更新对应类型的waiting_time_minutes(分钟)。"""
        num_chargers = int(getattr(self, f"num_chargers_{charger_type}"))  # factorial/range需要int，文件加载的数量按整数使用
        mean_service_time = self.mean_service_time_minutes  # μ，站点级别，两种类型共用
        service_time_std = self.service_time_std_minutes    # σ，站点级别，两种类型共用
        offered_load = self._offered_load(charger_type)

        if offered_load <= 0 or mean_service_time <= 0 or num_chargers <= 0:
            waiting_time = 0.0
        else:
            if offered_load >= num_chargers:
                # 违反稳定性条件 c > ρ，保守截断避免除零/发散
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
            waiting_time = prefactor * probability_of_wait

        setattr(self, f"waiting_time_minutes_{charger_type}", waiting_time)


class StationManager:
    """站点集合组件"""

    def __init__(self, stations_config: Optional[list[dict[str, Any]]] = None) -> None:
        # stations_config是站点参数的列表，每一项是一个字典（对应一份JSON式的站点配置，
        # 字段名要跟ChargingStation.__init__的形参名一一对应），站点对象在这里才真正创建，
        # 不是外部先建好、这里只做转存。
        self._stations: dict[str, ChargingStation] = {}  # key为站点id，value为站点对象

        for station_config in stations_config or []:
            station = ChargingStation(**station_config)
            if station.station_id in self._stations:
                raise ValueError(f"充电站 ID 已存在: {station.station_id}")

            self._stations[station.station_id] = station

    # ------------------------------------------------------------------
    # 公共接口：reset / step / get_state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """重置re所有站点。"""
        for station in self._stations.values():
            station.reset()

    def step(self, current_time: float, tou_tariff: float = 0.0) -> None:
        """把当前时间和分时电价转发给每个站点，让它们各自按自己的到达率时刻表刷新派生量。"""
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
