"""基础仿真端到端测试入口。"""
from typing import Any

from sim_env.core_env import EVChargingEnv
from sim_env.core_road_network import RoadNetwork
import sim_env.default_road_network as default_road_network

def main() -> None:
    # 加载数据
    road_matrix = default_road_network.road_matrix
    speed_matrix = default_road_network.speed_matrix
    speed_timetable = default_road_network.speed_timetable
    start_time = default_road_network.start_time
    # 创建环境对象
    road_network = RoadNetwork(
        matrix=road_matrix,
        speed_matrix=speed_matrix,
        speed_timetable=speed_timetable,
        start_time=start_time,
    )
    print(road_network.get_edge_between(0, 1))



def print_vehicle_state(label: str, observation: dict[str, Any]) -> None:
    """打印测试车辆和环境的关键状态。"""
    vehicle_state = observation["vehicles"]["vehicles"]["vehicle_001"]
    mobility_state = observation["mobility"]
    station_state = observation["stations"]

    print(
        label,
        {
            "time": observation["current_time"],
            "node": vehicle_state["current_node_id"],
            "status": vehicle_state["status"],
            "soc": round(vehicle_state["soc"], 4),
            "distance_km": round(vehicle_state["total_distance_km"], 4),
            "travel_time": round(vehicle_state["total_travel_time"], 2),
            "active_plan_count": mobility_state["active_plan_count"],
            "station_count": station_state["station_count"],
        },
    )


if __name__ == "__main__":
    main()
