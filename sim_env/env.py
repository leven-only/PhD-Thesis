"""环境整合模块，用于统一调度路网、车辆、站点和时间推进的仿真主循环。"""

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class EnvConfig:
    """环境运行的基础配置。"""

    start_time: float = 0.0
    end_time: float = 24 * 60 * 60
    time_step: float = 60.0
    random_seed: Optional[int] = 42


@dataclass(frozen=True)
class ObservationItem:
    """环境观测中的一个字段配置。"""

    key: str
    getter: Callable[["EVChargingEnv"], Any]


@dataclass
class EnvComponent:
    """环境组件注册项。"""

    name: str
    component: Any
    include_in_observation: bool = True


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
        mobility_manager: Optional[Any] = None,
        demand_generator: Optional[Any] = None,
        extra_components: Optional[dict[str, Any]] = None,
    ) -> None:
        # 形参设置
        self.config = config or EnvConfig()
        self.road_network = road_network
        self.station_manager = station_manager
        self.vehicle_manager = vehicle_manager
        self.mobility_manager = mobility_manager
        self.demand_generator = demand_generator

        # 其他设置
        self.current_time = self.config.start_time
        self.step_count = 0
        self.done = False
        self.metrics: dict[str, Any] = {}
        self.components: dict[str, EnvComponent] = {}
        self.observation_items: dict[str, ObservationItem] = {}

        # 组件 和 可观测信息 注册
        self._register_default_observations()
        self._register_default_components()

        # 对额外的组件进行注册
        for name, component in (extra_components or {}).items():
            self._register_component(name, component)

    # 集中统一管理环境自身可观测信息
    def _register_default_observations(self) -> None:
        """注册环境自身观测字段。"""
        self._register_observation("current_time", lambda env: env.current_time)
        self._register_observation("step_count", lambda env: env.step_count)
        self._register_observation("done", lambda env: env.done)

    # 集中统一管理环境组件，后期可以考虑把 所有组件放在一起，不分开注册
    def _register_default_components(self) -> None:
        """注册默认组件；reset、step 和 observation 都从这里读取。"""
        self._register_component("demand", self.demand_generator, include_in_observation=False)
        self._register_component("road_network", self.road_network)
        self._register_component("mobility", self.mobility_manager)
        self._register_component("vehicles", self.vehicle_manager)
        self._register_component("stations", self.station_manager)

    # 更新组件信息
    def _update_component(self, component: Optional[Any], action: Optional[Any] = None) -> None:
        """推进组件一个时间步。"""
        if component is None:
            return

        self._call_component_method(
            component,
            "step",
            self.config.time_step,
            self.current_time,
            action,
        )

    # 返回组件状态信息
    def _get_component_state(self, component: Optional[Any]) -> Any:
        """返回组件状态信息。"""
        if component is None:
            return None

        return self._call_component_method(component, "get_state")

    # 重置component
    def _reset_component(self, component: Optional[Any]) -> None:
        """重置组件。"""
        if component is None:
            return

        self._call_component_method(component, "reset")

    # 调用组件方法
    def _call_component_method(self, component: Any, method_name: str, *args: Any) -> Any:
        """调用组件方法；如果方法不存在或不可调用则报错。"""
        method = getattr(component, method_name, None)

        if not callable(method):
            component_type = type(component).__name__
            raise TypeError(
                f"组件 {component_type} 必须实现可调用方法: {method_name}()"
            )

        return method(*args)

    # 注册组件，顺便注册可观测
    def _register_component(
        self,
        name: str,
        component: Optional[Any],
        include_in_observation: bool = True,    # 决定这个状态是否可被观测
    ) -> None:
        """注册环境组件。"""
        if component is None:
            return

        self.components[name] = EnvComponent(
            name=name,
            component=component,
            include_in_observation=include_in_observation,
        )

        if include_in_observation:
            self._register_observation(
                name,
                lambda env, component_name=name: env._get_component_state(
                    env.components[component_name].component
                ),
            )

    # 注册一个可观测信息
    def _register_observation(
        self,
        key: str,
        getter: Callable[["EVChargingEnv"], Any],
    ) -> None:
        """注册环境观测字段。"""
        self.observation_items[key] = ObservationItem(key=key, getter=getter)

    # 启动环境，不断推进时间步
    def run(self, max_steps: Optional[int] = None) -> dict[str, Any]:
        """运行完整仿真，并返回最终统计指标。"""
        self.reset()

        while not self.done:
            if max_steps is not None and self.step_count >= max_steps:
                break

            self.step()

        return self.get_metrics()

    # 推进一个时间步，启动环境更新等工作
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

        for component_item in self.components.values():
            self._update_component(component_item.component, action=action)

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

    # 重置环境
    def reset(self) -> dict[str, Any]:
        """重置环境，并返回初始观测。"""
        self.current_time = self.config.start_time
        self.step_count = 0
        self.done = False
        self.metrics = {
            "total_steps": 0,
            "finished": False,
        }

        for component_item in self.components.values():
            self._reset_component(component_item.component)

        return self.get_observation()

    # 获得所有可观测环境的信息
    def get_observation(self) -> dict[str, Any]:
        """返回当前环境观测。"""
        return {
            item.key: item.getter(self)
            for item in self.observation_items.values()
        }

    # 计算当前时间步骤的奖励
    def compute_reward(self) -> float:
        """计算当前时间步奖励。

        初始版本不绑定具体优化目标，因此默认返回 0。
        后续可以替换为等待时间、绕行时间、充电失败率等指标的组合。
        """
        return 0.0

    # 返回统计信息
    def get_info(self) -> dict[str, Any]:
        """返回调试和统计信息。"""
        return {
            "current_time": self.current_time,
            "step_count": self.step_count,
            "time_step": self.config.time_step,
            "metrics": self.get_metrics(),
        }

    # 返回统计指标
    def get_metrics(self) -> dict[str, Any]:
        """返回环境累计统计指标。"""
        return dict(self.metrics)
