"""ChargingManager：负责应用充电方案，把方案里的充电时长/功率/费用作用到车辆上。

跟 MobilityManager 一样，本模块只负责"应用方案"，不管方案是怎么生成的（选哪个站、
充多少、费用多少都由上层算法算好塞进方案里）。充电方案是尽量通用的 dict，字段固定为
三个：duration_minutes（充电时长，分钟）、power_kw（充电功率，kW）、cost（充电费用，
上层算好的总价，这里不区分电费/服务费）。
"""

from copy import deepcopy
from typing import Any, Optional

from _02_sim_env.base_vehicle import Vehicle


class ChargingManager:
    """充电方案管理器：把一份充电方案应用到唯一一辆车上（单车调度场景）。"""

    def __init__(self, vehicle: Vehicle) -> None:
        """绑定 vehicle，方案相关的属性先初始化为空，等外部再下发方案。"""
        self.vehicle = vehicle  # 车辆对象，充电结果（充入电量/累计花费）作用到它身上

        self.plan: Optional[dict[str, Any]] = None  # 充电方案：含 duration_minutes/power_kw/cost 三个字段，没有方案时是 None

        # 构造完成时的初始值快照，供 reset() 还原用
        self.initial_state = {
            "plan": self.plan,
        }

    def reset(self) -> None:
        """把方案相关的属性都还原成构造完成时的初始值（清空当前充电方案）。"""
        for name, value in self.initial_state.items():
            setattr(self, name, deepcopy(value))

    def assign_plan(self, charging_plan: dict[str, Any]) -> None:
        """接收一个新的充电方案：先 reset() 清空上一个方案的残留，再校验字段并保存方案。

        charging_plan 的三个字段（键名固定）：
            duration_minutes: 充电时长（分钟）
            power_kw:         充电功率（kW）
            cost:             充电费用（上层算好的总价，不区分电费/服务费）
        """
        self.reset()

        required_fields = ("duration_minutes", "power_kw", "cost")
        missing_fields = [field for field in required_fields if field not in charging_plan]
        if missing_fields:
            raise ValueError(
                f"充电方案缺少字段: {missing_fields}，需要包含 {list(required_fields)}"
            )

        self.plan = dict(charging_plan)

    def step(self):
        """应用当前充电方案：按功率×时长换算充入电量，连同费用一起作用到车辆上，
        然后清空方案。充电方案是"一次 step 应用完"的，所以返回的 finished_flag 恒为 1。

        返回 (finished_flag, time_used)：
            finished_flag=1：充电方案已应用完，内部已 reset() 清空。
            time_used        ：这次充电实际占用的时间（分钟），等于方案里的 duration_minutes。
        """
        # 应用前先检查有没有方案：没有就说明 step() 前忘了调用 assign_plan()
        if self.plan is None:
            raise RuntimeError("没有充电方案，请先调用 assign_plan() 初始化 plan，再调用 step()")

        duration_minutes = float(self.plan["duration_minutes"])  # 充电时长（分钟）
        power_kw = float(self.plan["power_kw"])                  # 充电功率（kW）
        cost = float(self.plan["cost"])                          # 充电费用（上层算好的总价）

        # 充入电量(kWh) = 功率(kW) × 时长(小时)；时长是分钟，除以 60 转成小时
        energy_kwh = power_kw * duration_minutes / 60.0

        # 把算好的电量/费用作用到车辆上：车辆自己负责累加充入电量、增加 SOC、累计花费
        self.vehicle.charge(energy_kwh=energy_kwh, cost=cost)

        time_used = duration_minutes  # 充电占用的时间就是方案里的充电时长

        self.reset()  # 方案应用完，清空自己，等外部下一次 assign_plan()

        return 1, time_used
