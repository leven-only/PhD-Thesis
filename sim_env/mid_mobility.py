"""移动组件，负责执行路径计划、算出车辆移动结果，并直接写回车辆。

单车化重构（2026-09-08）：不再通过 VehicleManager 按 vehicle_id 索引多辆车，
构造时直接持有唯一的 Vehicle 对象；相应地，路径计划/移动结果也从"按
vehicle_id 建字典"简化成"只有一份"。
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

from sim_env.base_road_network import RoadNetwork
from sim_env.base_vehicle import Vehicle, VehicleStatus


@dataclass
class MobilityPlan:
    """车辆正在执行的路径计划（单车场景下全局只会有一份，或没有）。"""

    path: list[int]
    next_node_index: int = 1
    current_edge_remaining_km: Optional[float] = None


@dataclass
class MovementResult:
    """一次 step() 中的移动计算结果。"""

    distance_km: float = 0.0
    travel_time: float = 0.0
    current_node_id: Optional[int] = None  # 这一步结束时，最后完全到达的节点
    next_node_id: Optional[int] = None     # 这一步结束时，正朝向的下一个节点（可能还没到）
    edge_progress_km: float = 0.0          # 在current_node_id->next_node_id这条边上已经走了多远
    status: Optional[VehicleStatus] = None
    remove_plan: bool = False


@dataclass
class MobilityStepResult:
    """step() 的返回值，供 Env 决定这一步时钟该走多久：
    车辆状态如果变了（比如到达终点/没电/开始行驶），time_used 就是这一步
    实际消耗的时间；状态没变（比如本来就在DRIVING、这一步继续开、没有
    提前结束），time_used 就是完整的 time_step（Env 侧其实不看这个值，
    因为按下面 status_changed=False 分支会直接用 self.time_step，但
    这里仍然填一个真实值，避免调用方误用出错）。
    """

    status_changed: bool = False
    time_used: float = 0.0


class MobilityManager:
    """
    只负责计算"这一步车辆移动了多少"，并把结果直接应用到车辆上。

    单车场景：构造时直接接收唯一的 Vehicle 对象，不再需要 VehicleManager
    做按 vehicle_id 的索引/转发。
    """

    def __init__(
        self,
        road_network: RoadNetwork,
        vehicle: Vehicle,
    ) -> None:
        self.road_network = road_network
        self.vehicle = vehicle
        self._active_plan: Optional[MobilityPlan] = None

    # ------------------------------------------------------------------
    # 反应式组件接口：reset / step / get_state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """清空当前路径计划。"""
        self._active_plan = None

    def step(self, time_step: float, action: Optional[Any] = None) -> MobilityStepResult:
        """应用路径动作、推进当前的计划，把移动结果直接写回车辆，并报告这一步
        车辆状态是否发生了变化、实际用了多长时间——Env 用这个结果决定时钟这一步
        该走满 time_step，还是只走车辆实际用掉的时间（状态变化意味着车辆提前
        结束了这段时间预算，比如到达终点/没电，剩下的预算不该被时钟白白吃掉）。

        不看current_time：这一步用到的"路网当前边速度"这类跟时间有关的信息，
        应该是外部（Env）在调用这里之前先调用过road_network.step(current_time)
        刷新好的，Mobility自己只管"给定这段时间(time_step)，车辆能走多远"，不
        需要再关心现在几点，这跟Vehicle.step()不看时间是同一个道理。

        action 目前约定为形如 {"path": [...]} 的字典（单车场景下不再需要按
        vehicle_id 区分是哪辆车的路径）；没有新路径时传 None 或不含 "path"
        键即可，会继续推进上一步遗留的计划。
        """
        status_before = self.vehicle.status

        if isinstance(action, dict) and "path" in action:
            self._set_path(action["path"])

        if self._active_plan is None:
            # 要么本来就没有计划、要么_set_path()因为"单节点路径=已到终点"这种
            # 情况直接把状态置成了FINISHED并清空了计划——不管是哪种，这一步都
            # 没有发生任何"消耗时间的移动"，time_used如实报告成0。
            return MobilityStepResult(
                status_changed=self.vehicle.status != status_before,
                time_used=0.0,
            )

        result = self._calculate_movement(self.vehicle, self._active_plan, time_step)

        if result.remove_plan:
            self._active_plan = None

        self.vehicle.step(
            action="move",
            params={
                "distance_km": result.distance_km,
                "travel_time": result.travel_time,
                "current_node_id": result.current_node_id,
                "next_node_id": result.next_node_id,
                "edge_progress_km": result.edge_progress_km,
                "status": result.status,
            },
        )

        return MobilityStepResult(
            status_changed=self.vehicle.status != status_before,
            time_used=result.travel_time,
        )

    def get_state(self) -> dict[str, Any]:
        """返回当前路径执行状态。"""
        return {
            "has_active_plan": self._active_plan is not None,
            "active_plan": (
                None
                if self._active_plan is None
                else {
                    "path": deepcopy(self._active_plan.path),
                    "next_node_index": self._active_plan.next_node_index,
                    "current_edge_remaining_km": self._active_plan.current_edge_remaining_km,
                }
            ),
        }

    # ------------------------------------------------------------------
    # 内部实现（下划线开头，外部代码不应依赖）
    # ------------------------------------------------------------------

    def _set_path(self, path: list[int]) -> None:
        if not path:
            raise ValueError("路径不能为空")
        if self.vehicle.current_node_id != path[0]:
            raise ValueError(
                f"路径起点 {path[0]} 与车辆当前位置 "
                f"{self.vehicle.current_node_id} 不一致"
            )

        if len(path) == 1:
            if path[0] != self.vehicle.destination_node_id:
                raise ValueError("单节点路径只能用于已经到达终点的车辆")

            self._active_plan = None
            self.vehicle.step(
                action="set_status",
                params={"status": VehicleStatus.FINISHED},
            )
            return

        self._active_plan = MobilityPlan(path=list(path))

    def _calculate_movement(
        self,
        vehicle: Vehicle,
        plan: MobilityPlan,
        time_step: float,
    ) -> MovementResult:
        remaining_time = max(float(time_step), 0.0)
        remaining_energy_distance = vehicle.available_distance_km()
        result = MovementResult()
        reached_node_id: Optional[int] = None

        while remaining_time > 0 and self._can_move(vehicle):
            if plan.next_node_index >= len(plan.path):
                result.status = VehicleStatus.FINISHED
                result.remove_plan = True
                break

            current_node = plan.path[plan.next_node_index - 1]
            next_node = plan.path[plan.next_node_index]
            edge = self.road_network.get_edge_between(current_node, next_node)

            if plan.current_edge_remaining_km is None:
                plan.current_edge_remaining_km = edge["length_km"]

            # 注意：全局时间单位已统一为"分钟"（Env.time_step / RoadNetwork.step()
            # 的 current_time 都是分钟），所以这里把 speed_kph 换算成"公里/分钟"，
            # 不能再按旧写法换算成"公里/秒"再乘以 remaining_time——旧写法在
            # remaining_time 实际传入的是分钟时会把行驶距离放大60倍，这是2026-09-07
            # 全局改成分钟制时，mid_mobility.py 这份草稿没有同步到的一处遗留问题，
            # 这次顺手一并修正。
            speed_km_per_minute = max(edge["speed_kph"], 1e-6) / 60.0
            travel_distance = min(
                plan.current_edge_remaining_km,
                speed_km_per_minute * remaining_time,
                remaining_energy_distance,
            )

            if travel_distance <= 0:
                result.status = VehicleStatus.FAILED
                result.remove_plan = True
                break

            travel_time = travel_distance / speed_km_per_minute
            result.distance_km += travel_distance
            result.travel_time += travel_time
            remaining_time -= travel_time
            remaining_energy_distance -= travel_distance
            plan.current_edge_remaining_km -= travel_distance

            if remaining_energy_distance <= 1e-9:
                result.status = VehicleStatus.FAILED
                result.remove_plan = True
                break

            if plan.current_edge_remaining_km > 1e-9:
                break

            reached_node_id = next_node
            plan.next_node_index += 1
            plan.current_edge_remaining_km = None

        # 不管这一步是走完了一整条边、只走了半条边、还是完全没动，都从plan当前
        # 的状态（next_node_index/current_edge_remaining_km）把"现在具体在哪"
        # 算出来，每次都汇报current_node_id/next_node_id/edge_progress_km，
        # 不再是"只有到达完整节点才汇报位置"（这是原来代码的一个已知缺口）。
        if plan.next_node_index < len(plan.path):
            position_current_node = plan.path[plan.next_node_index - 1]
            position_next_node = plan.path[plan.next_node_index]
            edge = self.road_network.get_edge_between(position_current_node, position_next_node)
            remaining = plan.current_edge_remaining_km
            edge_progress_km = 0.0 if remaining is None else edge["length_km"] - remaining

            result.current_node_id = position_current_node
            result.next_node_id = position_next_node
            result.edge_progress_km = edge_progress_km
        else:
            # 已经到终点：next_node_id故意设成跟current_node_id一样（而不是None），
            # 是因为move()对"传None"的约定是"这个字段不变"，如果这里传None，
            # next_node_id会停留在到达终点前那一刻的旧值，变成一个过期数据。
            # 设成等于current_node_id，直观地表示"不再朝任何地方走"，同时避免
            # 改动已经定稿过的Vehicle.move()的参数语义。
            result.current_node_id = plan.path[-1]
            result.next_node_id = plan.path[-1]
            result.edge_progress_km = 0.0

        if reached_node_id == vehicle.destination_node_id:
            result.status = VehicleStatus.FINISHED
            result.remove_plan = True
        elif result.distance_km > 0 and result.status is None:
            result.status = VehicleStatus.DRIVING

        return result

    @staticmethod
    def _can_move(vehicle: Vehicle) -> bool:
        return vehicle.status not in (
            VehicleStatus.FINISHED,
            VehicleStatus.FAILED,
            VehicleStatus.CHARGING,
            VehicleStatus.QUEUEING,
        )
