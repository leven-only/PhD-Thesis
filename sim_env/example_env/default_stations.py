"""默认充电站样例。"""

from sim_env.core_station import ChargingStation, StationManager


def build_default_station_manager() -> StationManager:
    """创建与默认路网对应的两个充电站。"""
    return StationManager(
        stations=[
            ChargingStation(
                station_id="station_001",
                mapped_node="node_06",
                access_distance_km=0.0,
                chargers={
                    "fast": {
                        "num_chargers": 2,
                        "power_kw": 60.0,
                    },
                },
                tou_tariff={
                    "fast": 1.0,
                },
            ),
            ChargingStation(
                station_id="station_002",
                mapped_node="node_09",
                access_distance_km=0.0,
                chargers={
                    "fast": {
                        "num_chargers": 1,
                        "power_kw": 90.0,
                    },
                },
                tou_tariff={
                    "fast": 1.2,
                },
            ),
        ],
    )
