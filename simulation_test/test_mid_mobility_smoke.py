"""mid_mobility.py 单车化重构后的冒烟测试：只测移动性，不碰充电。

场景（3个节点、2条边的简单路网，speed_matrix只给一个恒定快照，避免时变速度
干扰测试）：
    node0 --10km,60kph--> node1 --5km,60kph--> node2
60kph换算成"公里/分钟" = 1 km/min，所以：
    - 第一条边(10km)需要10分钟走完
    - 第二条边(5km)需要5分钟走完
    - 全程15分钟、15公里

两个测试场景各自独立创建一个新的 EVChargingEnv 实例，互不影响。
"""

from sim_env.top_env import EVChargingEnv
from sim_env.base_vehicle import VehicleStatus


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
# 测试(a)：给车辆一条多节点路径，连续调用几次 step()，
# 验证 current_node_id/next_node_id/edge_progress_km/soc 逐步正确变化。
# ----------------------------------------------------------------------
def test_multi_step_progress() -> None:
    env = build_env(time_step=1.0)  # 每步只推进1分钟，远小于15分钟的全程

    prev_soc = env.vehicle.soc
    prev_distance = env.vehicle.total_distance_km

    # 第一步：同时下发路径
    env.step(action={"path": [0, 1, 2]})
    assert env.vehicle.current_node_id == 0, env.vehicle.current_node_id
    assert env.vehicle.next_node_id == 1, env.vehicle.next_node_id
    assert approx(env.vehicle.edge_progress_km, 1.0), env.vehicle.edge_progress_km
    assert env.vehicle.status == VehicleStatus.DRIVING, env.vehicle.status
    assert env.vehicle.soc < prev_soc, "soc应该已经下降"
    assert env.vehicle.total_distance_km > prev_distance, "累计里程应该已经增加"
    prev_soc = env.vehicle.soc
    prev_distance = env.vehicle.total_distance_km

    # 再走几步，不用再传path，mobility应自动接着上次的计划走
    for expected_progress_km in (2.0, 3.0, 4.0, 5.0):
        env.step()
        assert env.vehicle.current_node_id == 0, env.vehicle.current_node_id
        assert env.vehicle.next_node_id == 1, env.vehicle.next_node_id
        assert approx(env.vehicle.edge_progress_km, expected_progress_km), (
            expected_progress_km,
            env.vehicle.edge_progress_km,
        )
        assert env.vehicle.status == VehicleStatus.DRIVING
        assert env.vehicle.soc < prev_soc, "soc应该持续下降"
        assert env.vehicle.total_distance_km > prev_distance, "累计里程应该持续增加"
        prev_soc = env.vehicle.soc
        prev_distance = env.vehicle.total_distance_km

    print("test_multi_step_progress PASSED")
    print(
        "  当前时间:", env.current_time,
        "| current_node_id:", env.vehicle.current_node_id,
        "| next_node_id:", env.vehicle.next_node_id,
        "| edge_progress_km:", round(env.vehicle.edge_progress_km, 4),
        "| soc:", round(env.vehicle.soc, 6),
        "| total_distance_km:", round(env.vehicle.total_distance_km, 4),
    )


# ----------------------------------------------------------------------
# 测试(b)：time_step 足够车辆一次 step() 走完全程，
# 验证车辆状态正确变为 FINISHED、不报错、不多算（距离/耗电/耗时都不超过应有值）。
# ----------------------------------------------------------------------
def test_single_step_finishes_route() -> None:
    env = build_env(time_step=1000.0)  # 远大于全程所需的15分钟

    env.step(action={"path": [0, 1, 2]})

    assert env.vehicle.status == VehicleStatus.FINISHED, env.vehicle.status
    assert env.vehicle.current_node_id == 2, env.vehicle.current_node_id
    assert env.vehicle.next_node_id == 2, env.vehicle.next_node_id
    assert approx(env.vehicle.edge_progress_km, 0.0), env.vehicle.edge_progress_km

    # 不多算：全程恰好15km/15分钟，不应该因为time_step给了1000分钟预算就多走
    assert approx(env.vehicle.total_distance_km, 15.0), env.vehicle.total_distance_km
    assert approx(env.vehicle.total_travel_time, 15.0), env.vehicle.total_travel_time

    expected_energy_kwh = 15.0 * 0.18
    expected_soc = 1.0 - expected_energy_kwh / 60.0
    assert approx(env.vehicle.soc, expected_soc), (env.vehicle.soc, expected_soc)

    # mobility自己的活跃计划应该已经清空
    mobility_state = env.mobility.get_state()
    assert mobility_state["has_active_plan"] is False, mobility_state

    # 时钟推进：车辆这一步状态从IDLE变成了FINISHED，属于"状态变化"，所以
    # 时钟只走车辆实际用掉的15分钟，不会被time_step=1000这个远大于实际需要的
    # 预算带着凭空跳到1000分钟。
    assert approx(env.current_time, 15.0), env.current_time

    print("test_single_step_finishes_route PASSED")
    print(
        "  当前时间:", env.current_time,
        "| status:", env.vehicle.status.value,
        "| current_node_id:", env.vehicle.current_node_id,
        "| total_distance_km:", round(env.vehicle.total_distance_km, 4),
        "| total_travel_time:", round(env.vehicle.total_travel_time, 4),
        "| soc:", round(env.vehicle.soc, 6),
    )


if __name__ == "__main__":
    test_multi_step_progress()
    print()
    test_single_step_finishes_route()
