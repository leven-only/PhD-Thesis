"""模拟环境启动入口。"""

from sim_env.env import EVChargingEnv, EnvConfig
from sim_env.mobility import MobilityManager
from sim_env.road_network import RoadNetwork
from sim_env.vehicle import Vehicle, VehicleManager


def main():
    config = EnvConfig(
        start_time=0,
        end_time=3600,
        time_step=60,
    )

    road_network = RoadNetwork()
    vehicle_manager = VehicleManager(
        vehicles=[
            Vehicle(
                vehicle_id="vehicle_001",
                origin_node_id="node_01",
                destination_node_id="node_10",
            )
        ]
    )
    mobility_manager = MobilityManager(
        road_network=road_network,
        vehicle_manager=vehicle_manager,
        auto_plan_shortest_path=True,
        path_weight="time",
    )

    env = EVChargingEnv(
        config=config,
        road_network=road_network,
        vehicle_manager=vehicle_manager,
        mobility_manager=mobility_manager,
    )

    observation = env.reset()
    print_vehicle_state("初始状态", observation)

    stop_reason = "environment_done"

    while not env.done:
        observation, reward, done, info = env.step()
        print_vehicle_state(f"第 {env.step_count} 步", observation)

        vehicle_state = observation["vehicles"]["vehicles"]["vehicle_001"]
        if vehicle_state["status"] in ("finished", "failed"):
            stop_reason = f"vehicle_{vehicle_state['status']}"
            break

    print("停止原因:", stop_reason)


def print_vehicle_state(label, observation):
    vehicle_state = observation["vehicles"]["vehicles"]["vehicle_001"]
    mobility_state = observation["mobility"]

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
        },
    )


if __name__ == "__main__":
    main()
