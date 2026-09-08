import math
from bisect import bisect_right
from copy import deepcopy
from typing import Any, Optional


class ChargingStation:
    """单个充电站；站点级别的抽象状态，不追踪具体充电桩占用。"""

    def __init__(
        self,
        num_chargers: list[float],                   # 充电桩数量：[slow, fast]，来自站点物理数据文件
        power_kw: list[float],                        # 充电功率(kW)：[slow, fast]，来自站点物理数据文件
        arrival_rate_per_hour: list[list[float]],     # 到达率数值本身：[slow数组, fast数组]，每个数组长度24，第h位是"第h个整点小时"
                                                       # 对应的到达率(辆/小时)，数值含义不变；这24个值对应的是哪个时间段，由
                                                       # _minute_index() 用分钟时刻表查表决定，不再是"数组下标本身就是小时数"
        initial_time: float,                          # 单位：分钟（跟整个仿真环境的时钟单位统一）。内部通过
                                                       # _minute_index() 按分钟时刻表查到达率，不再要求外部先换算成小时
        station_id: str,                              # 站点id
        mapped_node: int,                             # 映射的路网节点
        access_distance_km: float,                    # 到映射节点的距离(km)
        mean_service_time_minutes: list[float],       # 平均服务时间(分钟)：[slow, fast]，与num_chargers/power_kw同样按充电桩类型区分，不再是站点级别共用值
        service_time_std_minutes: list[float],        # 服务时间标准差(分钟)：[slow, fast]，与num_chargers/power_kw同样按充电桩类型区分，不再是站点级别共用值
        marginal_cost_factor: float = 0.2,            # 边际成本因子β，全局共享常数，默认0.2（proposal 3.4.2经验取值）
    ) -> None:
        # ---- 与充电桩类型无关的参数 ----
        self.station_id = station_id                      # 站点id
        self.mapped_node = mapped_node                    # 映射的路网节点
        self.access_distance_km = access_distance_km      # 到映射节点的距离(km)
        self.marginal_cost_factor = marginal_cost_factor  # 边际成本因子β=0.2，全局共享，不分站点、不分充电桩类型
        self.tou_tariff = 0.0                              # 分时电价，环境信息，来自外部电价表，由StationManager逐time slot注入，构造时先占位
        self._initial_time = initial_time                  # 仿真起始时间，只在构造和reset()时使用，构造后不应再被外部修改

        # ---- slow类型充电桩参数 ----
        self.num_chargers_slow = num_chargers[0]                            # slow充电桩数量
        self.power_kw_slow = power_kw[0]                                    # slow充电功率(kW)
        self.mean_service_time_minutes_slow = mean_service_time_minutes[0]  # slow平均服务时间(分钟)，与num_chargers/power_kw同样按类型拆分，来自站点物理数据文件，不随仿真变化
        self.service_time_std_minutes_slow = service_time_std_minutes[0]    # slow服务时间标准差(分钟)，与num_chargers/power_kw同样按类型拆分，来自站点物理数据文件，不随仿真变化
        self._arrival_rate_timetable_slow = arrival_rate_per_hour[0]        # slow到达率时刻表，长度24，来自场景配置文件，不随仿真变化
        self.arrival_rate_per_hour_slow = self._arrival_rate_timetable_slow[self._minute_index(initial_time)]  # slow当前生效到达率，构造时取initial_time对应时段的值，由step()按current_time更新

        self.available_chargers_slow = 0.0                                  # slow当前空闲桩数，动态量，由_recompute_derived_state()更新
        self.dynamic_service_fee_slow = 0.0                                 # slow动态服务费，动态量，由_recompute_derived_state()更新
        self.total_price_slow = 0.0                                         # slow总价，动态量，由_recompute_derived_state()更新
        self.waiting_time_minutes_slow = 0.0                                # slow预计等待时间(分钟)，动态量，由_recompute_derived_state()更新

        # ---- fast类型充电桩参数 ----
        self.num_chargers_fast = num_chargers[1]                            # fast充电桩数量
        self.power_kw_fast = power_kw[1]                                    # fast充电功率(kW)
        self.mean_service_time_minutes_fast = mean_service_time_minutes[1]  # fast平均服务时间(分钟)，与num_chargers/power_kw同样按类型拆分，来自站点物理数据文件，不随仿真变化
        self.service_time_std_minutes_fast = service_time_std_minutes[1]    # fast服务时间标准差(分钟)，与num_chargers/power_kw同样按类型拆分，来自站点物理数据文件，不随仿真变化
        self._arrival_rate_timetable_fast = arrival_rate_per_hour[1]        # fast到达率时刻表，长度24，来自场景配置文件，不随仿真变化
        self.arrival_rate_per_hour_fast = self._arrival_rate_timetable_fast[self._minute_index(initial_time)]  # fast当前生效到达率，构造时取initial_time对应时段的值，由step()按current_time更新

        self.available_chargers_fast = 0.0                                  # fast当前空闲桩数，动态量，由_recompute_derived_state()更新
        self.dynamic_service_fee_fast = 0.0                                 # fast动态服务费，动态量，由_recompute_derived_state()更新
        self.total_price_fast = 0.0                                         # fast总价，动态量，由_recompute_derived_state()更新
        self.waiting_time_minutes_fast = 0.0                                # fast预计等待时间(分钟)，动态量，由_recompute_derived_state()更新

    # ------------------------------------------------------------------
    # 公共接口：reset / step / get_state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """回到_initial_time这个时刻，并直接调用一次step()完成初始化，因此reset()之后
        全部动态量都是按_initial_time、tou_tariff=0算出来的真实值，不是占位的0.0。
        """
        self.step(current_time=self._initial_time, tou_tariff=0.0)

    def step(self, current_time: float, tou_tariff: float = 0.0) -> None:
        """按当前分时电价、当前时段对应的到达率刷新，重新计算全部动态量。

        current_time 单位是分钟，按 _minute_index() 查到分钟时刻表里对应的下标，
        直接从构造时存好的时刻表里取值，不再做任何计算（时刻表里的数就是最终
        到达率本身）。
        """
        self.tou_tariff = tou_tariff
        minute_index = self._minute_index(current_time)
        self.arrival_rate_per_hour_slow = self._arrival_rate_timetable_slow[minute_index]
        self.arrival_rate_per_hour_fast = self._arrival_rate_timetable_fast[minute_index]
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

    # 到达率时刻表的24个分段边界（单位：分钟），对应0点、1点……23点整点时刻，
    # 是固定的全局常量（24个整点小时均匀切分一天），不需要外部传入配置——如果
    # 以后需要不均匀的分段，再考虑把它改成构造参数。跟 RoadNetwork.speed_timetable
    # 是同一种"时刻表 + bisect查找"的写法，_minute_index() 就是这里的
    # _system_time2speed_index() 对应版本。
    _ARRIVAL_RATE_TIMETABLE_MINUTES = [hour * 60 for hour in range(24)]  # [0, 60, 120, ..., 1380]

    @staticmethod
    def _minute_index(current_time: float) -> int:
        """把任意时刻（单位：分钟，允许跨天累计，如第2天9点=1980.0）换算成到达率
        时刻表（_ARRIVAL_RATE_TIMETABLE_MINUTES）里对应的下标，用于查到达率。

        做法：先用 current_time % 1440 把跨天的时刻折算回"一天以内的第几分钟"
        （1440=一天的分钟数），再用 bisect_right 在24个整点边界里找到当前时刻
        落在哪一段，取这一段的左端点下标——这跟 RoadNetwork 里
        _system_time2speed_index() 用 speed_timetable 做 bisect 查找是同一个
        套路，只是这里的时刻表是固定的24个整点小时，不是外部传入的任意时刻表。

        构造函数、reset()（通过 step()）、step() 三处都用这一个函数，避免同样
        的取模/查表逻辑写三份。
        """
        time_of_day_minutes = current_time % 1440.0
        return max(
            bisect_right(ChargingStation._ARRIVAL_RATE_TIMETABLE_MINUTES, time_of_day_minutes) - 1,
            0,
        )

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
        mean_service_time_minutes = getattr(self, f"mean_service_time_minutes_{charger_type}")  # 按充电桩类型取对应的平均服务时间，不再是站点级别共用值
        mean_service_time_hours = mean_service_time_minutes / 60  # 分钟转小时，与λ的"每小时"单位对齐
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
        mean_service_time = getattr(self, f"mean_service_time_minutes_{charger_type}")  # μ，按充电桩类型取值，不再是站点级别共用
        service_time_std = getattr(self, f"service_time_std_minutes_{charger_type}")    # σ，按充电桩类型取值，不再是站点级别共用
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

    def __init__(
        self,
        stations_config: Optional[list[dict[str, Any]]] = None,
        initial_time: float = 0.0,  # 单位：分钟，跟 Env 自己的时钟统一，由这里统一分发给
                                     # 每一个站点，保证所有站点起点时间一致；不需要（也不
                                     # 应该）由外部在每份 station_config 里各自塞一份。
    ) -> None:
        # stations_config是站点参数的列表，每一项是一个字典（对应一份JSON式的站点配置），
        # 字段名要跟下面显式列出的这些 keyword 一一对应；这份字典本身不需要包含
        # initial_time——它是上面单独的构造参数，统一注入给每个站点。
        # 站点对象在这里才真正创建，不是外部先建好、这里只做转存。
        self._stations: dict[str, ChargingStation] = {}  # key为站点id，value为站点对象

        for station_config in stations_config or []:
            # 不用 ChargingStation(**station_config) 这种解构调用——那样字段对不对得上
            # 全靠字典里的 key 名字，出错了也不容易一眼看出来。这里显式把每个字段单独
            # 取出来再传，虽然啰嗦，但哪个字段对应哪个形参一目了然。
            station = ChargingStation(
                num_chargers=station_config["num_chargers"],
                power_kw=station_config["power_kw"],
                arrival_rate_per_hour=station_config["arrival_rate_per_hour"],
                initial_time=initial_time,
                station_id=station_config["station_id"],
                mapped_node=station_config["mapped_node"],
                access_distance_km=station_config["access_distance_km"],
                mean_service_time_minutes=station_config["mean_service_time_minutes"],
                service_time_std_minutes=station_config["service_time_std_minutes"],
                marginal_cost_factor=station_config.get("marginal_cost_factor", 0.2),
            )
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
        """查询一个站点并返回副本，仅用于查询"""
        if station_id not in self._stations:
            raise ValueError(f"充电站不存在: {station_id}")
        return deepcopy(self._stations[station_id])
