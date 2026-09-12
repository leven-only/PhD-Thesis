"""对env + Dijkstra一次性规划的组合，做一次多组OD对的冒烟测试：车辆能不能从
起点走到终点，走完之后的位置/累计里程/累计耗电这些数字是否合理。
"""

from _02_sim_env.top_env import EVChargingEnv
from _02_sim_env.default_road_network import (
    road_network_matrix,
    road_network_speed_matrix,
    road_network_speed_timetable,
)
from _02_sim_env.default_stations import default_stations_config
from _03_algorithms.dijkstra_routing.dijkstra_algorithm import DijkstraRouting


# 测试用的多组起点/终点对，覆盖路网里几种不同的距离和跳数
OD_PAIRS = [
    (0, 9),
    (1, 7),
    (2, 4),
    (3, 8),
    (5, 0),
    (0, 1),  # 只有一条边，最简单的情况
]


def build_env(origin_node_id, destination_node_id):
    """用默认路网和默认站点配置，构造一个只跑一次行程用的EVChargingEnv。"""
    return EVChargingEnv(
        initial_time=480.0,
        road_network_matrix=road_network_matrix,
        road_network_speed_matrix=road_network_speed_matrix,
        road_network_speed_timetable=road_network_speed_timetable,
        vehicle_id="veh_smoke_test",
        vehicle_origin_node_id=origin_node_id,
        vehicle_destination_node_id=destination_node_id,
        vehicle_soc=1.0,
        stations_config=default_stations_config,
    )


def run_one_od_pair(origin_node_id, destination_node_id):
    """跑一次完整行程：构造env，用一次性Dijkstra规划并执行，检查车辆是否真的到达终点。"""
    env = build_env(origin_node_id, destination_node_id)
    env.reset()

    algorithm = DijkstraRouting(weight="length_km")
    algorithm.interaction(env)

    vehicle = env.get_vehicle()
    dynamic_state = vehicle.get_dynamic_state()
    static_state = vehicle.get_state()

    arrived = dynamic_state["current_node_id"] == destination_node_id
    print(
        f"OD({origin_node_id}->{destination_node_id}): "
        f"到达={arrived}, "
        f"里程={static_state['total_distance_km']:.2f}km, "
        f"耗电={static_state['total_energy_used_kwh']:.3f}kWh, "
        f"耗时={static_state['total_travel_time']:.2f}分钟, "
        f"剩余soc={dynamic_state['soc']:.4f}"
    )
    return arrived


def main():
    """依次跑完OD_PAIRS里的每一组起点终点，最后汇总有没有全部到达终点。"""
    results = []
    for origin_node_id, destination_node_id in OD_PAIRS:
        arrived = run_one_od_pair(origin_node_id, destination_node_id)
        results.append(arrived)

    failed_count = results.count(False)
    if failed_count == 0:
        print("全部OD对都成功到达终点，冒烟测试通过")
    else:
        raise AssertionError(f"有{failed_count}组OD对没有到达终点，冒烟测试失败")


if __name__ == "__main__":
    main()
