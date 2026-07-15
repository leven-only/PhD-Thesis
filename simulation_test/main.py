"""基础仿真端到端测试入口。"""

import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation_test.builder import build_test_env


def main() -> None:
    env, context = build_test_env()
    observation = env.reset()
    print_vehicle_state("初始状态", observation)

    initial_path = context.road_network.shortest_path(
        "node_01",
        "node_10",
        weight="time",
    )
    first_action = {
        "vehicle_paths": {
            "vehicle_001": initial_path,
        }
    }

    stop_reason = "environment_done"
    while not env.done:
        action = first_action if env.step_count == 0 else None
        observation, _, _, _ = env.step(action)
        print_vehicle_state(f"第 {env.step_count} 步", observation)

        vehicle_state = observation["vehicles"]["vehicles"]["vehicle_001"]
        if vehicle_state["status"] in ("finished", "failed"):
            stop_reason = f"vehicle_{vehicle_state['status']}"
            break

    print("停止原因:", stop_reason)


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
