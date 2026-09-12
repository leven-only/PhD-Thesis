"""环境整合模块：持有 RoadNetwork / StationManager / 唯一一辆 Vehicle 三个基础组件，
负责在构造时用外部传入的参数创建并初始化它们，并推进仿真时间。

场景范围：单车调度（不追求同时支持多车），但对任意单车调度问题都通用。
当前是最小可跑版本：不含 mobility/重规划触发器/需求生成/reward，这些留到后续再接入。

全局时间单位统一为"分钟"（一天=1440分钟）。RoadNetwork、ChargingStation、Vehicle
三个 base 组件现在全部按分钟理解 current_time，Env 不需要再做任何单位换算。

构造参数一律按"一对一"显式传入，不使用 **dict 解构——虽然更啰嗦，但不用跳到
别的地方找某个 key 对不对得上,可读性更好。stations_config 是唯一例外：站点
数量不固定，没法拆成一个个具名参数，只能用列表。
"""

from typing import Any, Optional

from sim_env.base_road_network import RoadNetwork
from sim_env.base_station import StationManager, ChargingStation
from sim_env.base_vehicle import Vehicle
from sim_env.mid_mobility import MobilityManager


class EVChargingEnv:
    """电动汽车充电仿真的主环境（单车调度场景）。"""

    def __init__(
        self,
        *,
        # ---- 时钟 ----
        initial_time: float,  # 单位：分钟，一天按1440分钟计。随机生成留给外部，
                               # Env 只接收结果，内部不做任何随机采样。
        time_step: float = 1.0,  # 单位：分钟，每次 step() 最多推进的时长（见 step() 说明，
                                  # 这是"上限"，不是"每次一定精确走这么多"）。
        tou_tariff: float = 0.0,  # 分时电价，现在先用常数占位。

        # ---- RoadNetwork 的构造参数 ----
        road_network_matrix: Any,
        road_network_speed_matrix: Any,
        road_network_speed_timetable: Any,  # 单位：分钟，须跟 initial_time 用同一套时钟。

        # ---- 唯一一辆车的构造参数（单车调度场景，不需要 VehicleManager） ----
        vehicle_id: str,
        vehicle_origin_node_id: int,
        vehicle_destination_node_id: int,
        vehicle_soc: float,  # 出发时的初始电量(SOC比例)，Vehicle 自己也没给默认值，这里同样不给
        vehicle_battery_capacity_kwh: float = 60.0,
        vehicle_low_soc_threshold: float = 0.2,
        vehicle_target_soc: float = 0.8,
        vehicle_time_window_minutes: Optional[float] = None,  # 单位：分钟。
        vehicle_energy_consumption_kwh_per_km: float = 0.18,

        # ---- 站点集合的构造参数（数量不固定，只能用列表，其余参数都不用列表/字典） ----
        stations_config: list[dict[str, Any]],
    ) -> None:
        """
        stations_config: 每一项对应一个 ChargingStation 的构造参数。不需要（也不应该）
            包含 initial_time——同样由 Env 统一注入。ChargingStation 内部现在也是按
            "分钟"理解 initial_time/current_time（跟 Env 时钟单位一致，不用再换算），
            这里统一注入只是为了保证所有站点的起始时间跟 Env 自己的时钟对得上，避免
            外部不小心给不同站点传了不一致的起始时间。
        """
        self.time_step = time_step
        self.tou_tariff = tou_tariff

        self._initial_time = initial_time  # 分钟，reset() 用于回到这个起始时刻
        self.current_time = initial_time   # 分钟，仿真当前时间，允许跨天累计增长

        # 三个 基础 环境对象
        self.road_network = RoadNetwork(
            matrix=road_network_matrix,
            speed_matrix=road_network_speed_matrix,
            speed_timetable=road_network_speed_timetable,
            start_time=self.current_time,
        )

        self.station_manager = StationManager(
            stations_config=stations_config,
            initial_time=self.current_time,
        )

        self.vehicle = Vehicle(
            vehicle_id=vehicle_id,
            soc=vehicle_soc,
            battery_capacity_kwh=vehicle_battery_capacity_kwh,
            low_soc_threshold=vehicle_low_soc_threshold,
            origin_node_id=vehicle_origin_node_id,
            destination_node_id=vehicle_destination_node_id,
            target_soc=vehicle_target_soc,
            time_window_minutes=vehicle_time_window_minutes,
            energy_consumption_kwh_per_km=vehicle_energy_consumption_kwh_per_km,
        )

        self.mobility = MobilityManager(
            road_network=self.road_network,
            vehicle=self.vehicle,
        )

        # 构造完成后立刻按initial_time刷新一次road_network/station_manager：
        self.road_network.step(self.current_time)
        self.station_manager.step(
            current_time=self.current_time,
            tou_tariff=self.tou_tariff,
        )

    # ------------------------------------------------------------------
    # 基本三件套：reset / step / get_state
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        """把仿真时间和三个组件都还原到起始状态。"""
        self.current_time = self._initial_time

        self.road_network.reset()
        self.station_manager.reset()
        self.vehicle.reset()
        self.mobility.reset()

        return self.get_state()

    def step(self, action: Optional[Any] = None) -> dict[str, Any]:
        # (1) 有新方案就先下发：action里带"path"就是下发移动方案；不带"path"的
        # 分支现在先原样留空(pass)，将来是下发充电方案的地方(比如action里带
        # 类似charging_station_id这样的字段)。
        if isinstance(action, dict) and "path" in action:
            if self.mobility.path is not None:
                raise RuntimeError(
                    "当前还有方案没走完(mobility.path不是None)，不能再下发新方案；"
                    "要等上一次step()返回之后（此时方案必定已经走完），再传新的path。"
                )
            self.mobility.assign_plan(action["path"])
        else:
            pass  # 充电方案下发逻辑，稍后实现

        has_mobility_plan = self.mobility.path is not None
        has_charging_plan = False  # 充电还没实现，先占位；接入mid_charging.py后换成真正的判断

        # (2)+(3) mobility和charging互斥，各自独立地逐tick推进到自己的方案走完为止
        if has_mobility_plan:
            while True:
                # 这一tick能用多少时间：不超过"距离下一个tick边界"这段时间
                elapsed_in_tick = (self.current_time - self._initial_time) % self.time_step
                available_time = self.time_step - elapsed_in_tick

                finished_flag, time_used = self.mobility.step(available_time=available_time)

                self.current_time += time_used

                # 用推进后的新时间刷新环境：下一轮循环、以及方案走完后的返回值都要用最新状态
                self.road_network.step(self.current_time)
                self.station_manager.step(
                    current_time=self.current_time,
                    tou_tariff=self.tou_tariff,
                )

                if finished_flag == 1:
                    break  # 这次下发的移动方案已经走完，结束循环，返回

        elif has_charging_plan:
            # 充电逻辑，稍后实现。结构上应该跟mobility分支对称：自己的while循环，
            # 每一轮算available_time -> 调用充电模块的step()拿到(finished_flag,
            # time_used) -> self.current_time += time_used -> 刷新环境 ->
            # finished_flag==1就break。现在还没有充电模块，先占位。
            pass
        else:
            raise RuntimeError(
                "车辆既没有移动方案，也没有充电方案：不符合业务规则"
                "（任意时刻必须有且只有一个方案在跑，不能什么都不做）。"
            )

        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        """返回当前状态。暂时先把三个组件各自的 get_state() 结果原样汇总返回，
        以后如果算法需要的 observation 结构跟这个不一样，再单独调整。
        """
        return {
            "current_time": self.current_time,
            "road_network": self.road_network.get_state(),
            "stations": self.station_manager.get_state(),
            "vehicle": self.vehicle.get_state(),
        }

    # ------------------------------------------------------------------
    # 个性化查询接口：返回三个基础组件对象本身
    # ------------------------------------------------------------------

    def get_road_network(self) -> RoadNetwork:
        return self.road_network

    def get_station_manager(self) -> StationManager:
        return self.station_manager

    def get_station(self, station_id) -> ChargingStation:
        return self.station_manager.get_station(station_id=station_id)

    def get_vehicle(self) -> Vehicle:
        return self.vehicle
