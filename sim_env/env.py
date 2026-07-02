"""环境整合模块，用于统一调度路网、车辆、站点和时间推进的仿真主循环。"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class EnvConfig:
    """环境运行的基础配置。"""

    start_time: float = 0.0
    end_time: float = 24 * 60 * 60
    time_step: float = 60.0
    random_seed: Optional[int] = 42


class EVChargingEnv:
    """电动汽车充电仿真的主环境。

    这个类先定义环境的核心生命周期：
    reset -> step -> run。

    路网、车辆、站点和需求模块后续实现后，可以作为组件传入本环境。
    """

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        road_network: Optional[Any] = None,
        station_manager: Optional[Any] = None,
        vehicle_manager: Optional[Any] = None,
        demand_generator: Optional[Any] = None,
    ) -> None:
        self.config = config or EnvConfig()
        self.road_network = road_network
        self.station_manager = station_manager
        self.vehicle_manager = vehicle_manager
        self.demand_generator = demand_generator

        self.current_time = self.config.start_time
        self.step_count = 0
        self.done = False
        self.metrics: dict[str, Any] = {}

    def reset(self) -> dict[str, Any]:
        """重置环境，并返回初始观测。"""
        self.current_time = self.config.start_time
        self.step_count = 0
        self.done = False
        self.metrics = {
            "total_steps": 0,
            "finished": False,
        }

        self._reset_component(self.road_network)
        self._reset_component(self.station_manager)
        self._reset_component(self.vehicle_manager)
        self._reset_component(self.demand_generator)

        return self.get_observation()

    def step(self, action: Optional[Any] = None) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """推进一个仿真时间步。

        Parameters
        ----------
        action:
            预留给调度策略或强化学习策略的动作。

        Returns
        -------
        observation, reward, done, info
            初始版本使用简单 Gym 风格接口。
        """
        if self.done:
            return self.get_observation(), 0.0, True, self.get_info()

        self._update_component(self.demand_generator, action=action)
        self._update_component(self.vehicle_manager, action=action)
        self._update_component(self.station_manager, action=action)
        self._update_component(self.road_network, action=action)

        self.current_time += self.config.time_step
        self.step_count += 1

        if self.current_time >= self.config.end_time:
            self.done = True

        self.metrics["total_steps"] = self.step_count
        self.metrics["finished"] = self.done

        observation = self.get_observation()
        reward = self.compute_reward()
        info = self.get_info()

        return observation, reward, self.done, info

    def run(self, max_steps: Optional[int] = None) -> dict[str, Any]:
        """运行完整仿真，并返回最终统计指标。"""
        self.reset()

        while not self.done:
            if max_steps is not None and self.step_count >= max_steps:
                break

            self.step()

        return self.get_metrics()

    def get_observation(self) -> dict[str, Any]:
        """返回当前环境观测。"""
        return {
            "current_time": self.current_time,
            "step_count": self.step_count,
            "done": self.done,
            "road_network": self._get_component_state(self.road_network),
            "stations": self._get_component_state(self.station_manager),
            "vehicles": self._get_component_state(self.vehicle_manager),
        }

    def compute_reward(self) -> float:
        """计算当前时间步奖励。

        初始版本不绑定具体优化目标，因此默认返回 0。
        后续可以替换为等待时间、绕行时间、充电失败率等指标的组合。
        """
        return 0.0

    def get_info(self) -> dict[str, Any]:
        """返回调试和统计信息。"""
        return {
            "current_time": self.current_time,
            "step_count": self.step_count,
            "time_step": self.config.time_step,
            "metrics": self.get_metrics(),
        }

    def get_metrics(self) -> dict[str, Any]:
        """返回环境累计统计指标。"""
        return dict(self.metrics)

    def _reset_component(self, component: Optional[Any]) -> None:
        """如果组件实现了 reset 方法，则调用它。"""
        if component is not None and hasattr(component, "reset"):
            component.reset()

    def _update_component(self, component: Optional[Any], action: Optional[Any] = None) -> None:
        """如果组件实现了 step 或 update 方法，则推进它。"""
        if component is None:
            return

        if hasattr(component, "step"):
            component.step(self.config.time_step, self.current_time, action)
            return

        if hasattr(component, "update"):
            component.update(self.config.time_step, self.current_time, action)

    def _get_component_state(self, component: Optional[Any]) -> Any:
        """尽量以统一方式读取组件状态。"""
        if component is None:
            return None

        if hasattr(component, "get_state"):
            return component.get_state()

        if hasattr(component, "state"):
            return component.state

        return None
