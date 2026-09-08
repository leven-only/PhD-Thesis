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
        """推进一个时间步。

        (1) 车辆移动：mobility 这一步用到的道路速度/站点状态，用的是"最近一次
            环境刷新"留下的结果——可能是 reset() 构造时按 initial_time 刷新的，
            也可能是上一次 step() 结束时按那时的 current_time 刷新的；因为
            current_time 在两次刷新之间不会变，这里不需要再重复刷新一次
            （重复刷新会算出完全一样的结果，纯粹浪费）；
        (2) 时钟推进：如果车辆状态变了（到达终点/没电/开始行驶……），说明这段
            time_step 预算被提前用掉了一部分，用 mobility 实际报告的
            time_used 推进时钟，不把没用到的那部分预算凭空吃掉；状态没变
            就直接走满 self.time_step；
        (3) 用推进后的新时间刷新一次环境：一是让这次 step() 返回的观测快照里，
            road_network/stations 跟 current_time 对得上、不会慢半拍；二是
            顺便把这份状态留给下一次 step() 的第(1)步直接用，不用再刷新一遍。
        """
        # (1) 车辆移动
        mobility_result = self.mobility.step(time_step=self.time_step, action=action)

        # (2) 时钟推进：按车辆这一步是否发生状态变化，决定走多久
        if mobility_result.status_changed:
            self.current_time += mobility_result.time_used
        else:
            self.current_time += self.time_step

        # (3) 用推进后的新时间刷新环境
        self.road_network.step(self.current_time)
        self.station_manager.step(
            current_time=self.current_time,
            tou_tariff=self.tou_tariff,
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
