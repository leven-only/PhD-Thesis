"""模拟环境启动入口。"""

from sim_env.example_env.builder import build_default_env


def main():
    # 创建默认示例环境和组件引用。
    env, context = build_default_env()

    observation = env.reset()   # 重置环境，并 返回初始状态
    print_vehicle_state("初始状态", observation)

    # 由 road network提供最短路径
    initial_path = context.road_network.shortest_path(
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
    station_state = observation["stations"]

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
            "station_count": station_state["station_count"],
        },
    )


if __name__ == "__main__":
    main()
