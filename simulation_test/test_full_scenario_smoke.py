"""跑通一个稍微像"路网"的场景（4个节点、5条边，不是一条直线），覆盖当前
env.step()"一次调用把方案跑到底"这套设计下能想到的主要边缘情况。

路网结构（node0是出发点，node3是真正的终点；(0,2)和(1,3)是没在主路径上用到
的"旁路"，让路网不是一条直线，更接近真实情况）：

               6km            9km            3km
        node0 ------> node1 ------> node2 ------> node3
          \\                          ^
           \\ 20km(旁路，不用)        / 10km(旁路，不用)
            \\------------------------/
                     node0->node2                node1->node3

主路径 node0->node1->node2->node3 全长 6+9+3=18km。除非某个测试单独指定，
限速统一用60kph(=1km/min)，这样手算时间/距离都很直接。

覆盖的边缘情况：
    1. 3-4个节点的完整路径一次性下发，能正常跑完到终点(FINISHED)
    2. 路径终点不是车辆真正的目的地，走完后应该是IDLE不是FINISHED
    3. RL式的两节点短方案，分3次下发，累计走完整条路径
    4. 方案结束时间正好卡在tick边界上的边界情况（不多算/不少算）
    5. 方案还没走完就想下发新方案，必须被拒绝(RuntimeError)
    6. 一次调用里既没有移动方案也没有充电方案，必须报错(RuntimeError)
    7. 下发的路径起点和车辆当前位置不一致，必须报错(ValueError)
    8. 下发的路径只有1个节点（不够2个），必须报错(ValueError)
    9. 时间跨天累计增长(current_time超过1440)不应该报错
"""

from sim_env.top_env import EVChargingEnv


ROAD_NETWORK_MATRIX = [
    [0, 6, 20, 0],   # node0 -> node1(6km), node0 -> node2(20km，旁路)
    [0, 0, 9, 10],   # node1 -> node2(9km), node1 -> node3(10km，旁路)
    [0, 0, 0, 3],    # node2 -> node3(3km)
    [0, 0, 0, 0],    # node3是终点，没有出边
]
# 边的顺序按矩阵里非零元素从左到右、从上到下扫描得到：
#   (0,1)=6km, (0,2)=20km, (1,2)=9km, (1,3)=10km, (2,3)=3km
# 所以下面speed_matrix每一行要按这个顺序给5个速度值。
CONSTANT_SPEED_MATRIX = [[60.0, 60.0, 60.0, 60.0, 60.0]]  # 全部恒定60kph=1km/min
CONSTANT_SPEED_TIMETABLE = [0.0]


def build_env(time_step: float, initial_time: float = 0.0) -> EVChargingEnv:
    return EVChargingEnv(
        initial_time=initial_time,
        time_step=time_step,
        tou_tariff=0.0,
        road_network_matrix=ROAD_NETWORK_MATRIX,
        road_network_speed_matrix=CONSTANT_SPEED_MATRIX,
        road_network_speed_timetable=CONSTANT_SPEED_TIMETABLE,
        vehicle_id="vehicle_001",
        vehicle_origin_node_id=0,
        vehicle_destination_node_id=3,  # 真正的目的地是node3
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
# 1. 完整的4节点路径一次性下发，应该正常跑完，状态变成FINISHED
# ----------------------------------------------------------------------
def test_full_path_reaches_real_destination() -> None:
    env = build_env(time_step=2.0)  # 每个tick 2分钟，全程18分钟，会跑9个内部tick

    env.step(action={"path": [0, 1, 2, 3]})

    assert env.vehicle.current_node_id == env.vehicle.destination_node_id, env.vehicle.current_node_id
    assert env.vehicle.current_node_id == 3, env.vehicle.current_node_id
    assert env.vehicle.next_node_id == 3, env.vehicle.next_node_id
    assert approx(env.vehicle.edge_progress_km, 0.0), env.vehicle.edge_progress_km
    assert approx(env.vehicle.total_distance_km, 18.0), env.vehicle.total_distance_km
    assert approx(env.vehicle.total_travel_time, 18.0), env.vehicle.total_travel_time
    assert approx(env.current_time, 18.0), env.current_time

    expected_energy_kwh = 18.0 * 0.18
    expected_soc = 1.0 - expected_energy_kwh / 60.0
    assert approx(env.vehicle.soc, expected_soc), (env.vehicle.soc, expected_soc)
    assert env.mobility.path is None, env.mobility.path

    print("test_full_path_reaches_real_destination PASSED")


# ----------------------------------------------------------------------
# 2. 路径终点(node2)不是真正目的地(node3)，走完后应该是IDLE
# ----------------------------------------------------------------------
def test_partial_path_stops_at_idle() -> None:
    env = build_env(time_step=2.0)

    env.step(action={"path": [0, 1, 2]})  # 只走到node2，全程15km/15分钟

    assert env.vehicle.current_node_id != env.vehicle.destination_node_id, env.vehicle.current_node_id
    assert env.vehicle.current_node_id == 2, env.vehicle.current_node_id
    assert approx(env.vehicle.total_distance_km, 15.0), env.vehicle.total_distance_km
    assert approx(env.current_time, 15.0), env.current_time
    assert env.mobility.path is None, env.mobility.path

    print("test_partial_path_stops_at_idle PASSED")


# ----------------------------------------------------------------------
# 3. RL式：每次只给两个节点的短方案，分3次下发，走完整条18km路径
# ----------------------------------------------------------------------
def test_rl_style_three_segments_across_network() -> None:
    env = build_env(time_step=1000.0)  # 预算远大于每段所需时间，模拟"一步蹚到底"

    env.step(action={"path": [0, 1]})
    assert env.vehicle.current_node_id != env.vehicle.destination_node_id, env.vehicle.current_node_id
    assert approx(env.vehicle.total_distance_km, 6.0), env.vehicle.total_distance_km
    assert approx(env.current_time, 6.0), env.current_time

    env.step(action={"path": [1, 2]})
    assert env.vehicle.current_node_id != env.vehicle.destination_node_id, env.vehicle.current_node_id
    assert approx(env.vehicle.total_distance_km, 15.0), env.vehicle.total_distance_km
    assert approx(env.current_time, 15.0), env.current_time

    env.step(action={"path": [2, 3]})  # 这段到真正的终点
    assert env.vehicle.current_node_id == env.vehicle.destination_node_id, env.vehicle.current_node_id
    assert approx(env.vehicle.total_distance_km, 18.0), env.vehicle.total_distance_km
    assert approx(env.current_time, 18.0), env.current_time

    print("test_rl_style_three_segments_across_network PASSED")


# ----------------------------------------------------------------------
# 4. 方案结束时间正好卡在tick边界上：验证mobility.step()内部"预算恰好用完
#    同时路径也恰好走完"这个临界情况不会多算或者少算时间/距离，也不会需要
#    额外多调用一次才检测到方案结束（mid_mobility.py里在每次移动之后都会
#    立刻检查是否走到头，不需要等到下一轮循环开头才发现）。
#    node0->node1(6km)，time_step=3.0：第1个tick走3km(预算用完，方案没完)，
#    第2个tick再走3km，这次预算和路径同时耗尽，应该在这一次调用里就
#    正确识别为finished_flag=1。
# ----------------------------------------------------------------------
def test_plan_finishes_exactly_on_tick_boundary() -> None:
    env = build_env(time_step=3.0)

    env.step(action={"path": [0, 1]})  # 6km，恰好是2个tick(3km+3km)

    assert env.vehicle.current_node_id != env.vehicle.destination_node_id, env.vehicle.current_node_id  # node1不是终点
    assert env.vehicle.current_node_id == 1, env.vehicle.current_node_id
    assert approx(env.vehicle.edge_progress_km, 0.0), env.vehicle.edge_progress_km
    assert approx(env.vehicle.total_distance_km, 6.0), env.vehicle.total_distance_km
    assert approx(env.vehicle.total_travel_time, 6.0), env.vehicle.total_travel_time
    # 时间不多不少正好是6.0，不会因为"多转一圈"而多算
    assert approx(env.current_time, 6.0), env.current_time
    assert env.mobility.path is None, env.mobility.path

    print("test_plan_finishes_exactly_on_tick_boundary PASSED")


# ----------------------------------------------------------------------
# 5. 方案还没走完就想下发新方案：必须报错，不能静默顶替
#    (正常通过env.step()不会走到这个状态，因为step()返回时方案必定已经走完；
#     这里故意绕过env.step()，直接在mobility上assign_plan()来模拟"方案还
#     没走完"这个中间状态，专门测这个保护条件本身有没有生效)
# ----------------------------------------------------------------------
def test_cannot_reassign_plan_while_active() -> None:
    env = build_env(time_step=2.0)

    env.mobility.assign_plan([0, 1])  # 绕开env.step()，模拟"方案还在跑"的状态

    try:
        env.step(action={"path": [0, 1, 2]})
        raised = False
    except RuntimeError:
        raised = True

    assert raised, "方案还没走完时下发新方案应该报RuntimeError"
    print("test_cannot_reassign_plan_while_active PASSED")


# ----------------------------------------------------------------------
# 6. 既没有移动方案也没有充电方案：必须报错，车辆不能什么都不做
# ----------------------------------------------------------------------
def test_no_plan_at_all_raises() -> None:
    env = build_env(time_step=2.0)  # 刚构造完，mobility.path是None，没下发过任何方案

    try:
        env.step()  # 不带action
        raised = False
    except RuntimeError:
        raised = True

    assert raised, "既没有移动方案也没有充电方案时应该报RuntimeError"
    print("test_no_plan_at_all_raises PASSED")


# ----------------------------------------------------------------------
# 7. 路径起点和车辆当前位置不一致：必须报错
# ----------------------------------------------------------------------
def test_path_start_mismatch_raises() -> None:
    env = build_env(time_step=2.0)  # 车辆现在在node0

    try:
        env.step(action={"path": [1, 2]})  # 路径却是从node1开始
        raised = False
    except ValueError:
        raised = True

    assert raised, "路径起点跟车辆当前位置不一致时应该报ValueError"
    print("test_path_start_mismatch_raises PASSED")


# ----------------------------------------------------------------------
# 8. 路径只有1个节点：必须报错
# ----------------------------------------------------------------------
def test_path_too_short_raises() -> None:
    env = build_env(time_step=2.0)

    try:
        env.step(action={"path": [0]})
        raised = False
    except ValueError:
        raised = True

    assert raised, "路径只有1个节点时应该报ValueError"
    print("test_path_too_short_raises PASSED")


# ----------------------------------------------------------------------
# 9. 时间跨天累计增长(current_time超过1440)不应该报错
# ----------------------------------------------------------------------
def test_time_accumulates_across_day_boundary() -> None:
    env = build_env(time_step=5.0, initial_time=1430.0)  # 起始时间只比一天(1440)少10分钟

    env.step(action={"path": [0, 1, 2, 3]})  # 全程18分钟，会跨过1440这个"天"边界

    assert env.vehicle.current_node_id == env.vehicle.destination_node_id, env.vehicle.current_node_id
    assert approx(env.current_time, 1448.0), env.current_time  # 1430+18，没有被截断或出错
    assert approx(env.vehicle.total_distance_km, 18.0), env.vehicle.total_distance_km

    print("test_time_accumulates_across_day_boundary PASSED")


if __name__ == "__main__":
    tests = [
        test_full_path_reaches_real_destination,
        test_partial_path_stops_at_idle,
        test_rl_style_three_segments_across_network,
        test_plan_finishes_exactly_on_tick_boundary,
        test_cannot_reassign_plan_while_active,
        test_no_plan_at_all_raises,
        test_path_start_mismatch_raises,
        test_path_too_short_raises,
        test_time_accumulates_across_day_boundary,
    ]
    for test_fn in tests:
        test_fn()
    print()
    print(f"全部 {len(tests)} 个场景通过。")
