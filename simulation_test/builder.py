"""基础测试环境组装函数。"""

from dataclasses import dataclass
from typing import Optional

from sim_env.core_env import EVChargingEnv, EnvConfig
from sim_env.core_mobility import MobilityManager
from sim_env.core_road_network import RoadNetwork
from sim_env.core_station import StationManager
from sim_env.core_vehicle import Vehicle, VehicleManager
from sim_env.default_road_network import build_default_road_network
from sim_env.default_stations import build_default_station_manager


@dataclass
class TestEnvContext:
    """测试代码需要直接查询的组件引用。"""

    road_network: RoadNetwork
    vehicle_manager: VehicleManager
    mobility_manager: MobilityManager
    station_manager: StationManager


def build_test_env(
    config: Optional[EnvConfig] = None,
) -> tuple[EVChargingEnv, TestEnvContext]:
    """组装默认测试环境。"""
    env_config = config or EnvConfig(
        start_time=0,
        end_time=3600,
        time_step=60,
    )
    road_network = build_default_road_network()
    vehicle_manager = VehicleManager(
        vehicles=[
            Vehicle(
                vehicle_id="vehicle_001",
                origin_node_id=0,
                destination_node_id=9,
            )
        ]
    )
    mobility_manager = MobilityManager(
        road_network=road_network,
        vehicle_manager=vehicle_manager,
    )
    station_manager = build_default_station_manager()

    env = EVChargingEnv(
        config=env_config,
        road_network=road_network,
        station_manager=station_manager,
        vehicle_manager=vehicle_manager,
        mobility_manager=mobility_manager,
    )
    context = TestEnvContext(
        road_network=road_network,
        vehicle_manager=vehicle_manager,
        mobility_manager=mobility_manager,
        station_manager=station_manager,
    )
    return env, context
