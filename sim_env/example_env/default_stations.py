"""默认充电站样例。"""

from sim_env.station import ChargingStation, StationManager
from sim_env.vehicle import VehicleManager


def build_default_station_manager(vehicle_manager: VehicleManager) -> StationManager:
    """创建与默认路网对应的两个充电站。"""
    return StationManager(
        vehicle_manager=vehicle_manager,
        stations=[
            ChargingStation(
                station_id="station_001",
                node_id="node_06",
                charger_count=2,
                charging_power_kw=60.0,
                price_per_kwh=1.0,
            ),
            ChargingStation(
                station_id="station_002",
                node_id="node_09",
                charger_count=1,
                charging_power_kw=90.0,
                price_per_kwh=1.2,
            ),
        ],
    )
