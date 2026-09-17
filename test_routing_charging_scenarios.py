"""路径规划 + 充电优化的多方案测试。

用默认路网（10 节点、32 条边）和默认站点（station_001@node5、station_002@node8）
跑 4 个完整方案，每个方案都包含"路径规划（Dijkstra 最短路）+ 充电决策"，并在
模拟环境上执行、打印结果、用 assert 校验，方便手动对照。

统一设定：
    - 电池容量 60 kWh，能耗 0.18 kWh/km
    - 充电费用单价按占位值 1.0 元/kWh 计（费用 = 充入电量 × 单价），由"上层"
      （也就是本测试脚本）算好塞进 charging_plan，环境不参与定价。
    - station_001(node5)：fast 桩 60 kW；station_002(node8)：fast 桩 90 kW。
    - initial_time = 480 分钟（早 8 点）。

注意两个"耗时"的区别：
    - vehicle.total_travel_time 只累计"行驶"时间，不含充电。
    - env.current_time - initial_time 才是"行驶 + 充电"的总耗时。

运行方式：在项目根目录执行  python test_routing_charging_scenarios.py
"""

import networkx as nx

from _02_sim_env.top_env import EVChargingEnv
from _02_sim_env.default_road_network import (
    road_network_matrix,
    road_network_speed_matrix,
    road_network_speed_timetable,
)
from _02_sim_env.default_stations import default_stations_config


# 充电费用占位单价（元/kWh），实际应由上层按电价表/服务费算出后传入。
PRICE_PER_KWH = 1.0
# 充电站映射的路网节点与 fast 桩功率（与 default_stations.py 保持一致）。
STATION_001_NODE = 5   # fast 60 kW
STATION_002_NODE = 8   # fast 90 kW
TARGET_SOC = 0.8       # 充电目标电量


def make_env(origin, destination, soc):
    """构造一个只跑一次行程的环境，初始 SOC 可调。"""
    return EVChargingEnv(
        initial_time=480.0,
        road_network_matrix=road_network_matrix,
        road_network_speed_matrix=road_network_speed_matrix,
        road_network_speed_timetable=road_network_speed_timetable,
        vehicle_id="v",
        vehicle_origin_node_id=origin,
        vehicle_destination_node_id=destination,
        vehicle_soc=soc,
        vehicle_battery_capacity_kwh=60.0,
        stations_config=default_stations_config,
    )


def shortest_path(env, source, target):
    """在路网图上做一次 Dijkstra 最短路（按边长 length_km），返回节点路径。"""
    return nx.dijkstra_path(env.road_network.graph, source, target, weight="length_km")


def walk_path(env, path):
    """沿 path 走完整条方案（每次 step 走一条边，直到方案走完）。"""
    env.step({"path": path})
    while env.has_active_plan():
        env.step(None)


def charge_to(env, target_soc, power_kw):
    """在车辆当前位置充电到 target_soc：算好充电时长/费用后下发充电方案。
    返回 (充入电量, 充电时长分钟, 充电费用)。"""
    capacity = env.vehicle.battery_capacity_kwh
    energy_kwh = max(target_soc - env.vehicle.soc, 0.0) * capacity
    duration_minutes = energy_kwh / power_kw * 60.0
    cost = energy_kwh * PRICE_PER_KWH
    env.step({
        "charging_plan": {
            "duration_minutes": duration_minutes,
            "power_kw": power_kw,
            "cost": cost,
        }
    })
    return energy_kwh, duration_minutes, cost


def report(env, label):
    """打印一趟行程的关键结果，供手动对照。"""
    v = env.vehicle
    print(
        f"[{label}] 终点={v.current_node_id}  SOC={v.soc:.4f}  "
        f"里程={v.total_distance_km:.3f}km  行驶耗时={v.total_travel_time:.3f}min  "
        f"总耗时={env.current_time - 480.0:.3f}min  "
        f"充入={v.total_energy_charged_kwh:.3f}kWh  花费={v.total_cost:.3f}元"
    )


# ----------------------------------------------------------------------
# 方案 1：纯路径规划（不充电，作为基准对照）
# ----------------------------------------------------------------------
def scenario_1_baseline():
    """【场景】0 -> 9，初始 SOC = 1.0，电量充足，不需要充电。

    【路径规划】Dijkstra 最短路，预期路径 [0, 4, 8, 9]（0->4->8->9）。
    【充电优化】无（SOC 足够，全程不充电）。
    【预期结果】
        - 到达节点 9
        - 总里程 = 1.0 + 1.4 + 1.1 = 3.5 km
        - 耗电 3.5 × 0.18 = 0.63 kWh，最终 SOC = 1.0 - 0.63/60 = 0.9895
        - 充入 0 kWh，花费 0 元
    """
    env = make_env(0, 9, soc=1.0)
    env.reset()

    path = shortest_path(env, 0, 9)
    print(f"方案1 Dijkstra 路径 = {path}")
    walk_path(env, path)
    report(env, "方案1")

    v = env.vehicle
    assert path == [0, 4, 8, 9]
    assert v.current_node_id == 9
    assert abs(v.total_distance_km - 3.5) < 1e-6
    assert abs(v.soc - 0.9895) < 1e-6
    assert abs(v.total_energy_charged_kwh) < 1e-9
    assert abs(v.total_cost) < 1e-9


# ----------------------------------------------------------------------
# 方案 2：顺路充电（node8 就在最短路上，不绕道）
# ----------------------------------------------------------------------
def scenario_2_charge_on_route():
    """【场景】0 -> 9，初始 SOC = 0.05（人为低电量，模拟必须中途补电）。

    【路径规划】最短路 [0, 4, 8, 9] 本身途经 node8，而 node8 就是
        station_002，所以去 node8 充电"不绕道"。
    【充电优化】在 node8 用 fast 桩（90 kW）充到 target_soc 0.8：
        - 先走到 node8（0->4->8 = 2.4 km），耗电 0.432 kWh，SOC 降到 0.0428；
        - 充入 = (0.8 - 0.0428) × 60 = 45.432 kWh；
        - 时长 = 45.432 / 90 × 60 = 30.288 min；
        - 费用 = 45.432 元。
    【预期结果】
        - 到达节点 9，总里程 3.5 km（没绕道，和最短路一样长）
        - 充入 45.432 kWh，花费 45.432 元
        - 充完再走 8->9 耗 0.198 kWh，最终 SOC = 0.8 - 0.198/60 = 0.7967
        - 总耗时 = 行驶 6.4 + 充电 30.288 = 36.688 min
    """
    env = make_env(0, 9, soc=0.05)
    env.reset()

    path_to_station = shortest_path(env, 0, STATION_002_NODE)
    print(f"方案2 先到 node8 路径 = {path_to_station}")
    walk_path(env, path_to_station)

    energy, duration, cost = charge_to(env, TARGET_SOC, power_kw=90.0)
    print(f"方案2 在 node8 充电：充入 {energy:.3f} kWh / {duration:.3f} min / {cost:.3f} 元")

    walk_path(env, shortest_path(env, STATION_002_NODE, 9))
    report(env, "方案2")

    v = env.vehicle
    assert path_to_station == [0, 4, 8]
    assert v.current_node_id == 9
    assert abs(v.total_distance_km - 3.5) < 1e-6
    assert abs(v.total_energy_charged_kwh - 45.432) < 1e-3
    assert abs(v.total_cost - 45.432) < 1e-3
    assert abs(v.soc - 0.7967) < 1e-3


# ----------------------------------------------------------------------
# 方案 3：绕道充电（去 node5，需要多走路，作为方案 2 的对比）
# ----------------------------------------------------------------------
def scenario_3_charge_detour():
    """【场景】0 -> 9，初始 SOC = 0.05，跟方案 2 同样的低电量。

    【路径规划】最短路 [0, 4, 8, 9] 不经过 node5，所以去 node5（station_001）
        充电要绕道：0->4->5（2.2 km），充完再 5->9（1.6 km）。
    【充电优化】在 node5 用 fast 桩（60 kW）充到 0.8：
        - 先走到 node5 耗电 2.2 × 0.18 = 0.396 kWh，SOC 降到 0.0434；
        - 充入 = (0.8 - 0.0434) × 60 = 45.396 kWh；
        - 时长 = 45.396 / 60 × 60 = 45.396 min；
        - 费用 = 45.396 元。
    【预期结果】
        - 到达节点 9，总里程 = 2.2 + 1.6 = 3.8 km（比方案 2 的 3.5 km 多 0.3 km，绕道代价）
        - 充入 45.396 kWh，花费 45.396 元
        - 最终 SOC = 0.8 - (5->9 耗电 0.288/60) = 0.7952
        - 充电时长 45.396 min，明显比方案 2 的 30.288 min 长（60 kW 比 90 kW 慢）
    """
    env = make_env(0, 9, soc=0.05)
    env.reset()

    path_to_station = shortest_path(env, 0, STATION_001_NODE)
    print(f"方案3 先到 node5 路径 = {path_to_station}")
    walk_path(env, path_to_station)

    energy, duration, cost = charge_to(env, TARGET_SOC, power_kw=60.0)
    print(f"方案3 在 node5 充电：充入 {energy:.3f} kWh / {duration:.3f} min / {cost:.3f} 元")

    walk_path(env, shortest_path(env, STATION_001_NODE, 9))
    report(env, "方案3")

    v = env.vehicle
    assert path_to_station == [0, 4, 5]
    assert v.current_node_id == 9
    assert abs(v.total_distance_km - 3.8) < 1e-6
    assert abs(v.total_energy_charged_kwh - 45.396) < 1e-3
    assert abs(v.total_cost - 45.396) < 1e-3
    assert abs(v.soc - 0.7952) < 1e-3


# ----------------------------------------------------------------------
# 方案 4：长行程顺路充电（不同 OD，node5 在最短路上）
# ----------------------------------------------------------------------
def scenario_4_long_trip():
    """【场景】1 -> 7，初始 SOC = 0.05，换一个 OD 验证通用性。

    【路径规划】最短路 [1, 5, 6, 7] 途经 node5（station_001），所以 node5
        顺路，不绕道。
    【充电优化】在 node5 用 fast 桩（60 kW）充到 0.8：
        - 先走到 node5（1->5 = 1.0 km），耗电 0.18 kWh，SOC 降到 0.047；
        - 充入 = (0.8 - 0.047) × 60 = 45.180 kWh；
        - 时长 = 45.180 / 60 × 60 = 45.180 min；
        - 费用 = 45.180 元。
    【预期结果】
        - 到达节点 7，总里程 = 1.0 + 1.1 + 1.2 = 3.3 km
        - 充入 45.180 kWh，花费 45.180 元
        - 最终 SOC = 0.8 - (5->6->7 耗电 0.414/60) = 0.7931
    """
    env = make_env(1, 7, soc=0.05)
    env.reset()

    path_to_station = shortest_path(env, 1, STATION_001_NODE)
    print(f"方案4 先到 node5 路径 = {path_to_station}")
    walk_path(env, path_to_station)

    energy, duration, cost = charge_to(env, TARGET_SOC, power_kw=60.0)
    print(f"方案4 在 node5 充电：充入 {energy:.3f} kWh / {duration:.3f} min / {cost:.3f} 元")

    walk_path(env, shortest_path(env, STATION_001_NODE, 7))
    report(env, "方案4")

    v = env.vehicle
    assert path_to_station == [1, 5]
    assert v.current_node_id == 7
    assert abs(v.total_distance_km - 3.3) < 1e-6
    assert abs(v.total_energy_charged_kwh - 45.180) < 1e-3
    assert abs(v.total_cost - 45.180) < 1e-3
    assert abs(v.soc - 0.7931) < 1e-3


def main():
    print("=" * 72)
    scenario_1_baseline()
    print("-" * 72)
    scenario_2_charge_on_route()
    print("-" * 72)
    scenario_3_charge_detour()
    print("-" * 72)
    scenario_4_long_trip()
    print("=" * 72)
    print("全部方案通过（4 个方案的 assert 校验均成功）")


if __name__ == "__main__":
    main()
