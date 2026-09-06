from copy import deepcopy
from enum import Enum
from typing import Any, Optional


class VehicleStatus(str, Enum):
    """车辆状态。"""

    IDLE = "idle"
    DRIVING = "driving"
    SEEKING_CHARGE = "seeking_charge"
    QUEUEING = "queueing"
    CHARGING = "charging"
    FINISHED = "finished"
    FAILED = "failed"


class Vehicle:
    """单辆车的基础模型：只负责"车辆基础属性 + 基础属性变更"，不含任何路径/
    充电决策逻辑。不用dataclass，改成普通类+显式__init__，字段仍按物理含义分
    四类（静态属性/需求属性/动态属性/累积记录）摆放，方便对照阅读。

    __init__的参数全部是关键字参数（用了 * 强制），这样即使有些字段有默认值、
    有些没有，也不用被迫按"没默认值的参数必须排在有默认值的参数前面"这条规则
    把参数顺序打乱分类，可以完全按分类摆放（参考core_station.py同样的思路，
    也跟simulation_test/builder.py里已经是关键字调用的写法一致）。
    """

    def __init__(
        self,
        *,
        vehicle_id: str,
        battery_capacity_kwh: float = 60.0,
        low_soc_threshold: float = 0.2,
        origin_node_id: int,
        destination_node_id: int,
        target_soc: float = 0.8,
        time_window_minutes: Optional[float] = None,
        energy_consumption_kwh_per_km: float = 0.18,
        current_node_id: Optional[int] = None,
        next_node_id: Optional[int] = None,
        edge_progress_km: float = 0.0,
        soc: float = 1.0,
        status: VehicleStatus = VehicleStatus.IDLE,
        total_distance_km: float = 0.0,
        total_energy_used_kwh: float = 0.0,
        total_energy_charged_kwh: float = 0.0,
        total_travel_time: float = 0.0,
        total_cost: float = 0.0,
    ) -> None:
        # ---- 静态属性（车辆物理属性，一旦创建就不再变化）----
        self.vehicle_id = vehicle_id                        # 车辆id，必填，没有合理默认值
        self.battery_capacity_kwh = battery_capacity_kwh    # 电池总容量(kWh)
        self.low_soc_threshold = low_soc_threshold          # 低电量阈值(SOC比例)，低于此值视为需要优先寻找充电站

        # ---- 需求属性（这一次出行的需求参数：出发前设定，出行过程中本身不再变化，
        #      但属于"这一趟行程"而不是"这辆车"的属性，单独归一类）----
        self.origin_node_id = origin_node_id                # 出发节点，必填，没有合理默认值
        self.destination_node_id = destination_node_id      # 目的节点，必填，没有合理默认值
        self.target_soc = target_soc                        # 到达终点时的预期SOC
        self.time_window_minutes = time_window_minutes      # 本次出行的时间窗(分钟)：从origin到destination允许的总时长；None表示不设约束

        # ---- 动态属性（随仿真推进而变化的车辆状态量，由move()/charge()更新）----
        self.energy_consumption_kwh_per_km = energy_consumption_kwh_per_km  # 单位里程能耗(kWh/km)
        self.current_node_id = current_node_id if current_node_id is not None else origin_node_id  # 当前所在节点；不传时用origin_node_id填充
        self.next_node_id = next_node_id                    # 下一个目标节点；车辆正在某条边上行驶时才有意义
        self.edge_progress_km = edge_progress_km            # 在current_node_id->next_node_id这条边上已经走过的距离(km)
        self.soc = soc                                      # 当前电量(SOC比例)
        self.status = status                                # 当前状态

        # ---- 累积记录（仿真过程中只增不减的统计量；reset()时清零，回到构造完成时的初始值）----
        self.total_distance_km = total_distance_km                  # 累计行驶里程(km)
        self.total_energy_used_kwh = total_energy_used_kwh          # 累计耗电量(kWh)
        self.total_energy_charged_kwh = total_energy_charged_kwh    # 累计充入电量(kWh)
        self.total_travel_time = total_travel_time                  # 累计行驶时间
        self.total_cost = total_cost                                # 累计花费

        # 构造完成后对全部字段（不止get_state()暴露的那几个）做一次完整快照，
        # 供reset()还原成"创建对象时的状态"用。
        self._initial_state = self._full_state_snapshot()

    # ------------------------------------------------------------------
    # 反应式组件接口：reset / step / get_state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """恢复到创建对象时的完整状态（覆盖全部字段，不止get_state()返回的那几个）。"""
        for name, value in self._initial_state.items():
            setattr(self, name, deepcopy(value))

    def step(self, action: str, params: Optional[dict[str, Any]] = None) -> None:
        """根据action类型，把params转发给对应的方法，由step()统一负责判断
        "这一步到底是移动、充电、还是纯状态切换"，调用方（外部负责算物理结果
        的组件，比如替代Mobility的那个组件、Station）不需要自己决定该调用
        move()还是charge()，只要把算好的结果连同action标签一起传进来即可。

        不看current_time/time_step：车辆自己不关心"现在几点"，那是路网/站点
        这类"世界状态"组件的职责；车辆只关心"这一步外部告诉我发生了什么"，
        所以这里跟RoadNetwork.step(current_time)/ChargingStation.step(current_time, ...)
        的签名故意不一样。
        """
        params = params or {}
        if action == "move":
            self.move(**params)
        elif action == "charge":
            self.charge(**params)
        elif action == "set_status":
            # 纯状态切换（比如从DRIVING变成QUEUEING），不伴随移动或充电，
            # 所以不适合塞进move()/charge()里，单独给一个分支。
            self.status = VehicleStatus(params["status"])
        else:
            raise ValueError(f"未知动作类型: {action}，只能是\"move\"/\"charge\"/\"set_status\"")

    def get_state(self) -> dict[str, Any]:
        """返回车辆的静态属性 + 累积记录：车辆本身固定不变的身份/物理信息，
        加上整个仿真过程里的统计结果。不含需求属性（这次行程的参数，见
        get_demand_info()）和动态属性（当前位置/电量/状态，见get_dynamic_state()）。
        """
        return {
            "vehicle_id": self.vehicle_id,
            "battery_capacity_kwh": self.battery_capacity_kwh,
            "low_soc_threshold": self.low_soc_threshold,
            "total_distance_km": self.total_distance_km,
            "total_energy_used_kwh": self.total_energy_used_kwh,
            "total_energy_charged_kwh": self.total_energy_charged_kwh,
            "total_travel_time": self.total_travel_time,
            "total_cost": self.total_cost,
        }

    # ------------------------------------------------------------------
    # 信息查询接口：查询这次行程的需求参数、以及由当前状态推算出的信息
    # ------------------------------------------------------------------

    def get_demand_info(self) -> dict[str, Any]:
        """查询这一趟行程的需求参数：起点/终点/目标SOC/时间窗，出行过程中不变。"""
        return {
            "origin_node_id": self.origin_node_id,
            "destination_node_id": self.destination_node_id,
            "target_soc": self.target_soc,
            "time_window_minutes": self.time_window_minutes,
        }

    def available_distance_km(self) -> float:
        """查询当前电量对应的理论可行驶距离(km)，由soc/battery_capacity_kwh/
        energy_consumption_kwh_per_km现算得出，不是存储值。
        """
        if self.energy_consumption_kwh_per_km <= 0:
            return float("inf")

        available_energy_kwh = self.soc * self.battery_capacity_kwh
        return available_energy_kwh / self.energy_consumption_kwh_per_km

    # ------------------------------------------------------------------
    # 状态查询接口：查询随仿真推进变化的动态属性
    # ------------------------------------------------------------------

    def get_dynamic_state(self) -> dict[str, Any]:
        """查询车辆当前的动态属性：位置、电量、状态。"""
        return {
            "current_node_id": self.current_node_id,
            "next_node_id": self.next_node_id,
            "edge_progress_km": self.edge_progress_km,
            "soc": self.soc,
            "status": self.status.value,
        }

    # ------------------------------------------------------------------
    # 基础属性变更接口：由外部组件直接调用，不经过事件对象或step()中转
    # ------------------------------------------------------------------

    def move(
        self,
        distance_km: float = 0.0,
        travel_time: float = 0.0,
        current_node_id: Optional[int] = None,
        next_node_id: Optional[int] = None,
        edge_progress_km: Optional[float] = None,
        status: Optional[VehicleStatus] = None,
    ) -> None:
        """应用一步移动结果：更新位置和里程，并按车辆自己的能耗模型扣电。

        调用方（Mobility）只需要告诉车辆"这一步走了多少公里、现在在哪条边
        的什么位置"，具体耗多少电由车辆自己算，调用方不需要懂能耗模型。
        """
        distance_km = max(float(distance_km), 0.0)
        self.total_distance_km += distance_km
        self.total_travel_time += max(float(travel_time), 0.0)

        if current_node_id is not None:
            self.current_node_id = current_node_id
        if next_node_id is not None:
            self.next_node_id = next_node_id
        if edge_progress_km is not None:
            self.edge_progress_km = edge_progress_km
        if status is not None:
            self.status = VehicleStatus(status)

        if distance_km > 0:
            self._consume_energy(distance_km)

    def charge(
        self,
        energy_kwh: float = 0.0,
        cost: float = 0.0,
        status: Optional[VehicleStatus] = None,
    ) -> None:
        """应用一步充电结果：按充入的电量增加SOC、累计花费。

        跟移动不同，充电这一步"充了多少电"是充电桩功率和时长决定的物理
        结果，由充电组件算好直接传进来，车辆只负责应用。
        """
        energy_kwh = max(float(energy_kwh), 0.0)
        self.total_energy_charged_kwh += energy_kwh
        self.total_cost += max(float(cost), 0.0)

        if self.battery_capacity_kwh > 0:
            self.soc = min(self.soc + energy_kwh / self.battery_capacity_kwh, 1.0)
        if status is not None:
            self.status = VehicleStatus(status)

    # ------------------------------------------------------------------
    # 内部实现（下划线开头，外部代码不应依赖）
    # ------------------------------------------------------------------

    def _consume_energy(self, distance_km: float) -> None:
        """车辆自己的能耗模型：按行驶距离折算耗电量，扣减SOC。"""
        energy_used_kwh = distance_km * self.energy_consumption_kwh_per_km
        self.total_energy_used_kwh += energy_used_kwh

        if self.battery_capacity_kwh > 0:
            self.soc = max(self.soc - energy_used_kwh / self.battery_capacity_kwh, 0.0)

    def _full_state_snapshot(self) -> dict[str, Any]:
        """给reset()用的完整状态快照，覆盖全部字段（不止get_state()暴露的那几个），
        这样reset()才能真正恢复到"创建对象时的状态"，而不只是恢复get_state()里的字段。
        """
        return {
            "vehicle_id": self.vehicle_id,
            "battery_capacity_kwh": self.battery_capacity_kwh,
            "low_soc_threshold": self.low_soc_threshold,
            "origin_node_id": self.origin_node_id,
            "destination_node_id": self.destination_node_id,
            "target_soc": self.target_soc,
            "time_window_minutes": self.time_window_minutes,
            "energy_consumption_kwh_per_km": self.energy_consumption_kwh_per_km,
            "current_node_id": self.current_node_id,
            "next_node_id": self.next_node_id,
            "edge_progress_km": self.edge_progress_km,
            "soc": self.soc,
            "status": self.status,
            "total_distance_km": self.total_distance_km,
            "total_energy_used_kwh": self.total_energy_used_kwh,
            "total_energy_charged_kwh": self.total_energy_charged_kwh,
            "total_travel_time": self.total_travel_time,
            "total_cost": self.total_cost,
        }


class VehicleManager:
    """车辆集合组件；负责驱动车辆reset/step。"""

    def __init__(self, vehicles_config: Optional[list[dict[str, Any]]] = None) -> None:
        # 跟StationManager一样：接收JSON式的车辆参数列表，在这里用Vehicle(**vehicle_config)
        # 真正创建车辆对象，不接收外部预先建好的Vehicle对象。
        self._vehicles: dict[str, Vehicle] = {}  # key为车辆id，value为车辆对象

        for vehicle_config in vehicles_config or []:
            vehicle = Vehicle(**vehicle_config)
            if vehicle.vehicle_id in self._vehicles:
                raise ValueError(f"车辆 ID 已存在: {vehicle.vehicle_id}")

            self._vehicles[vehicle.vehicle_id] = vehicle

    # ------------------------------------------------------------------
    # 反应式组件接口：reset / step / get_state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """重置所有车辆。"""
        for vehicle in self._vehicles.values():
            vehicle.reset()

    def step(self, actions: Optional[dict[str, dict[str, Any]]] = None) -> None:
        """把每辆车对应的动作转发给它自己的step()。

        actions的key是vehicle_id，value是形如{"action": "move"/"charge"/"set_status",
        "params": {...}}的一份指令；具体由谁（替代Mobility的组件、Station等）算出
        这份指令，VehicleManager这里只按vehicle_id分发，不关心指令内容本身。
        """
        for vehicle_id, action_payload in (actions or {}).items():
            vehicle = self._vehicles.get(vehicle_id)
            if vehicle is None:
                raise ValueError(f"车辆不存在: {vehicle_id}")

            # 不用**action_payload这种字典解包的写法，改成显式取key，
            # 这样如果调用方漏传"action"，报错会直接是KeyError加上明确的
            # 提示信息，而不是step()那边比较生硬的"缺少参数"报错，读代码
            # 的人也能一眼看出action_payload里到底要有哪几个key。
            action = action_payload["action"]
            params = action_payload.get("params")
            vehicle.step(action=action, params=params)

    def get_state(self) -> dict[str, Any]:
        """返回车辆集合状态。"""
        status_counts: dict[str, int] = {}
        for vehicle in self._vehicles.values():
            status = vehicle.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "vehicle_count": len(self._vehicles),
            "status_counts": status_counts,
            "vehicles": {
                vehicle_id: vehicle.get_state()
                for vehicle_id, vehicle in self._vehicles.items()
            },
        }

    # ------------------------------------------------------------------
    # 个性化查询接口
    # ------------------------------------------------------------------

    def get_vehicle(self, vehicle_id: str) -> Vehicle:
        """查询一个车辆并返回副本，仅用于查询"""
        return deepcopy(self._vehicles[vehicle_id])
