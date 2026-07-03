"""模拟环境启动入口。"""

from sim_env.env import EVChargingEnv, EnvConfig
from sim_env.mobility import MobilityManager
from sim_env.road_network import RoadNetwork
from sim_env.vehicle import Vehicle, VehicleManager


def main():
    # 环境配置【后期将配置移动到专用配置文件】
    config = EnvConfig(
        start_time=0,
        end_time=3600,
        time_step=60,
    )

    # 建立路网对象
    road_network = RoadNetwork()

    # 建立汽车对象
    vehicle_manager = VehicleManager(
        vehicles=[
            Vehicle(
                vehicle_id="vehicle_001",   # 编号
                origin_node_id="node_01",   # 起点
                destination_node_id="node_10",  # 终点
            )
        ]
    )

    # 建立移动管理，负责汽车的移动
    mobility_manager = MobilityManager(
        road_network=road_network,  # 上传路网
        vehicle_manager=vehicle_manager,    # 将 汽车对象 交给 移动管理对象
    )

    # 建立 环境对象
    env = EVChargingEnv(
        config=config,  # 上传配置
        road_network=road_network,  # 上传路网
        vehicle_manager=vehicle_manager,    # 上传 汽车管理
        mobility_manager=mobility_manager,  # 上传 移动管理
    )

    observation = env.reset()   # 重置环境，并 返回初始状态
    print_vehicle_state("初始状态", observation)

    # 由 road network提供最短路径
    initial_path = road_network.shortest_path(
        "node_01",
        "node_10",
        weight="time",
    )
    first_action = {    # 第一组动作
        "vehicle_paths": {
            "vehicle_001": initial_path,
        }
    }

    stop_reason = "environment_done"    # 预设 停止原因
    while not env.done:
        action = first_action if env.step_count == 0 else None
        observation, reward, done, info = env.step(action)  # 每次执行一次step，step_count +1
        print_vehicle_state(f"第 {env.step_count} 步", observation)

        vehicle_state = observation["vehicles"]["vehicles"]["vehicle_001"]  # 更新汽车状态
        if vehicle_state["status"] in ("finished", "failed"):   # 当汽车状态时 finished，或者汽车 failed 结束执行
            stop_reason = f"vehicle_{vehicle_state['status']}"
            break

    print("停止原因:", stop_reason)


def print_vehicle_state(label, observation):
    vehicle_state = observation["vehicles"]["vehicles"]["vehicle_001"]
    mobility_state = observation["mobility"]

    # 这个函数需要更智能，不需要显示打印参数
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
