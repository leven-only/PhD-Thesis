"""mid_mobility.py / top_env.py 单车化重构后的冒烟测试：只测移动性，不碰充电。

env.step()现在的规则是："一次调用把当前下发的方案一路推进到方案结束才返回"，
调用方不用自己写while循环——每次调用都必须带一个新方案（除非上一次调用的方案
还没走完，但正常用法下不会出现这种情况，因为每次调用都会跑到方案结束）。

公共场景（3个节点、2条边的简单路网）：
    node0 --10km,60kph--> node1 --5km,60kph--> node2
60kph换算成"公里/分钟" = 1 km/min，所以：
    - 第一条边(10km)需要10分钟走完
    - 第二条边(5km)需要5分钟走完
    - 全程15分钟、15公里
测试(a)单独用了一个限速会中途变化的场景（验证env.step()内部确实逐tick刷新
了环境），其余测试都用上面这个恒定限速的公共场景。

各测试场景各自独立创建一个新的 EVChargingEnv 实例，互不影响。
"""

from sim_env.top_env import EVChargingEnv


ROAD_NETWORK_MATRIX = [
    [0, 10, 0],
    [0, 0, 5],
    [0, 0, 0],
]
ROAD_NETWORK_SPEED_MATRIX = [[60.0, 60.0]]  # 只有一个快照，对应两条边(0->1, 1->2)
ROAD_NETWORK_SPEED_TIMETABLE = [0.0]  # 单一快照，任何时刻都用它


def build_env(time_step: float) -> EVChargingEnv:
    return EVChargingEnv(
        initial_time=0.0,
        time_step=time_step,
        tou_tariff=0.0,
        road_network_matrix=ROAD_NETWORK_MATRIX,
        road_network_speed_matrix=ROAD_NETWORK_SPEED_MATRIX,
        road_network_speed_timetable=ROAD_NETWORK_SPEED_TIMETABLE,
        vehicle_id="vehicle_001",
        vehicle_origin_node_id=0,
        vehicle_destination_node_id=2,
        vehicle_soc=1.0,
        vehicle_battery_capacity_kwh=60.0,
        vehicle_low_soc_threshold=0.2,
        vehicle_target_soc=0.8,
        vehicle_time_window_minutes=None,
        vehicle_energy_consumption_kwh_per_km=0.18,
        stations_config=[],
    )


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 测试(a)：给车辆一条较长的路径（time_step=1.0，逐分钟tick），中途路网限速
# 会变化。这个测试验证的是env.step()内部"一个tick一个tick地推进+每个tick都
# 刷新环境"这件事真的在发生，而不是只在方案一开始按当时的限速算到底：
#   node0 --10km--> node1，限速在 t=0~5分钟是60kph(=1km/min)，
#                          t=5分钟往后降到30kph(=0.5km/min)。
#   如果每个tick都正确刷新了环境：前5分钟走5km，后面5km再用10分钟才能走完，
#       全程共15分钟、10km。
#   如果没有逐tick刷新（只在方案开始时按60kph算了一次）：10km会被错误地算成
#       只需要10分钟。
# 两者结果不同，可以直接通过总耗时/总里程区分开来。
# ----------------------------------------------------------------------
def test_long_plan_refreshes_speed_mid_route() -> None:
    variable_speed_matrix = [
        [60.0, 60.0],  # t=0起生效：两条边都是60kph
        [30.0, 60.0],  # t=5起生效：第一条边(0->1)降到30kph，第二条边不受影响
    ]
    variable_speed_timetable = [0.0, 5.0]

    env = EVChargingEnv(
        initial_time=0.0,
        time_step=1.0,  # 每个tick只有1分钟，全程要走很多个tick
        tou_tariff=0.0,
        road_network_matrix=ROAD_NETWORK_MATRIX,
        road_network_speed_matrix=variable_speed_matrix,
        road_network_speed_timetable=variable_speed_timetable,
        vehicle_id="vehicle_001",
        vehicle_origin_node_id=0,
        vehicle_destination_node_id=2,  # 真正的终点是node2，这次只走到node1，不是终点
        vehicle_soc=1.0,
        vehicle_battery_capacity_kwh=60.0,
        vehicle_low_soc_threshold=0.2,
        vehicle_target_soc=0.8,
        vehicle_time_window_minutes=None,
        vehicle_energy_consumption_kwh_per_km=0.18,
        stations_config=[],
    )

    # 只下发一次方案(0->1)，一次step()调用内部会自己逐tick推进直到方案走完
    env.step(action={"path": [0, 1]})

    assert env.vehicle.current_node_id == 1, env.vehicle.current_node_id
    assert env.vehicle.next_node_id == 1, env.vehicle.next_node_id
    assert approx(env.vehicle.edge_progress_km, 0.0), env.vehicle.edge_progress_km
    assert approx(env.vehicle.total_distance_km, 10.0), env.vehicle.total_distance_km

    # 关键断言：耗时是15分钟(5分钟60kph + 10分钟30kph)，不是10分钟
    # （10分钟意味着全程都被当成60kph处理了，说明中途没有正确刷新环境）
    assert approx(env.vehicle.total_travel_time, 15.0), env.vehicle.total_travel_time
    assert approx(env.current_time, 15.0), env.current_time

    # node1不是真正的终点，所以方案走完后车辆位置不应该等于目的地
    assert env.vehicle.current_node_id != env.vehicle.destination_node_id, env.vehicle.current_node_id
    assert env.mobility.path is None, env.mobility.path

    print("test_long_plan_refreshes_speed_mid_route PASSED")
    print(
        "  当前时间:", env.current_time,
        "| current_node_id:", env.vehicle.current_node_id,
        "| total_distance_km:", round(env.vehicle.total_distance_km, 4),
        "| total_travel_time:", round(env.vehicle.total_travel_time, 4),
    )

# 测试(b)：time_step 足够车辆一次 step() 走完全程，
# 验证车辆状态正确变为 FINISHED、不报错、不多算（距离/耗电/耗时都不超过应有值）。
# ----------------------------------------------------------------------
def test_single_step_finishes_route() -> None:
    env = build_env(time_step=1000.0)  # 远大于全程所需的15分钟

    env.step(action={"path": [0, 1, 2]})

    assert env.vehicle.current_node_id == env.vehicle.destination_node_id, env.vehicle.current_node_id
    assert env.vehicle.current_node_id == 2, env.vehicle.current_node_id
    assert env.vehicle.next_node_id == 2, env.vehicle.next_node_id
    assert approx(env.vehicle.edge_progress_km, 0.0), env.vehicle.edge_progress_km

    # 不多算：全程恰好15km/15分钟，不应该因为time_step给了1000分钟预算就多走
    assert approx(env.vehicle.total_distance_km, 15.0), env.vehicle.total_distance_km
    assert approx(env.vehicle.total_travel_time, 15.0), env.vehicle.total_travel_time

    expected_energy_kwh = 15.0 * 0.18
    expected_soc = 1.0 - expected_energy_kwh / 60.0
    assert approx(env.vehicle.soc, expected_soc), (env.vehicle.soc, expected_soc)

    # mobility自己的活跃方案应该已经清空（现在直接是mobility.path这个属性）
    assert env.mobility.path is None, env.mobility.path

    # 时钟推进：这一步mobility确实执行了一个方案(had_plan=True)，所以时钟只走
    # 车辆实际用掉的15分钟，不会被time_step=1000这个远大于实际需要的预算带着
    # 凭空跳到1000分钟。
    assert approx(env.current_time, 15.0), env.current_time

    print("test_single_step_finishes_route PASSED")
    print(
        "  当前时间:", env.current_time,
        "| current_node_id:", env.vehicle.current_node_id,
        "| total_distance_km:", round(env.vehicle.total_distance_km, 4),
        "| total_travel_time:", round(env.vehicle.total_travel_time, 4),
        "| soc:", round(env.vehicle.soc, 6),
    )


# ----------------------------------------------------------------------
# 测试(c)：RL式增量下发——外部每次只给"当前节点->下一个节点"两节点的短方案，
# 反复调用；验证：
#   1. 半截路径（不是真正终点）被走完后，状态是IDLE不是FINISHED，
#      时钟按实际耗时推进（不是time_step）；
#   2. 连续下发两段短方案，能正确累加时间、最终在真正到达终点时变成FINISHED。
# ----------------------------------------------------------------------
def test_rl_style_incremental_segments() -> None:
    env = build_env(time_step=1000.0)  # 预算远大于单段所需时间，模拟"一步蹚到底"

    # 第一段：0->1，只是全程的一半，不是真正的终点(2)
    env.step(action={"path": [0, 1]})
    assert env.vehicle.current_node_id != env.vehicle.destination_node_id, env.vehicle.current_node_id
    assert env.vehicle.current_node_id == 1, env.vehicle.current_node_id
    assert env.vehicle.next_node_id == 1, env.vehicle.next_node_id
    assert approx(env.vehicle.edge_progress_km, 0.0), env.vehicle.edge_progress_km
    assert approx(env.vehicle.total_distance_km, 10.0), env.vehicle.total_distance_km
    # 时钟只走了这一段实际耗时的10分钟，不是time_step=1000
    assert approx(env.current_time, 10.0), env.current_time
    assert env.mobility.path is None, "半截方案走完后应该清空，等待下一段指令"

    # 第二段：1->2，这次是真正的终点
    env.step(action={"path": [1, 2]})
    assert env.vehicle.current_node_id == env.vehicle.destination_node_id, env.vehicle.current_node_id
    assert env.vehicle.current_node_id == 2, env.vehicle.current_node_id
    assert env.vehicle.next_node_id == 2, env.vehicle.next_node_id
    assert approx(env.vehicle.total_distance_km, 15.0), env.vehicle.total_distance_km
    assert approx(env.vehicle.total_travel_time, 15.0), env.vehicle.total_travel_time
    # 两段时间正确累加：10 + 5 = 15，不是10 + 1000
    assert approx(env.current_time, 15.0), env.current_time
    assert env.mobility.path is None, env.mobility.path

    print("test_rl_style_incremental_segments PASSED")
    print(
        "  当前时间:", env.current_time,
        "| current_node_id:", env.vehicle.current_node_id,
        "| total_distance_km:", round(env.vehicle.total_distance_km, 4),
        "| total_travel_time:", round(env.vehicle.total_travel_time, 4),
    )


if __name__ == "__main__":
    test_long_plan_refreshes_speed_mid_route()
    print()
    test_single_step_finishes_route()
    print()
    test_rl_style_incremental_segments()
