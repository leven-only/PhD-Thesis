"""基础仿真测试使用的默认路网数据：10 个节点、32 条有向道路的邻接矩阵 +
单一速度快照。

三个变量名特意加了 road_network_ 前缀，跟 EVChargingEnv.__init__() 的
road_network_matrix / road_network_speed_matrix / road_network_speed_timetable
三个形参一一对应，调用处直接按名字传，不用怀疑传参顺序或者名字对不对得上：

    from _02_sim_env.default_road_network import (
        road_network_matrix,
        road_network_speed_matrix,
        road_network_speed_timetable,
    )
    env = EVChargingEnv(
        ...,
        road_network_matrix=road_network_matrix,
        road_network_speed_matrix=road_network_speed_matrix,
        road_network_speed_timetable=road_network_speed_timetable,
        ...,
    )

注意：这里不包含 start_time——RoadNetwork 的 start_time 由 EVChargingEnv
在构造时自动用它自己的 initial_time 传入（见 top_env.py），不是路网数据的
一部分，也不需要外部在这里重复提供一份。
"""

# 邻接矩阵：0 表示没有道路，非零值表示道路长度（km）。10 个节点，共 32 条有向边。
road_network_matrix = [
    [0.0, 1.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.2, 0.0, 1.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.1, 0.0, 1.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.3, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 1.4, 0.0],
    [0.0, 1.0, 0.0, 0.0, 1.2, 0.0, 1.1, 0.0, 1.0, 1.6],
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.1, 0.0, 1.2, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.2, 0.0, 0.0, 1.4],
    [0.0, 0.0, 0.0, 0.0, 1.4, 1.0, 0.0, 0.0, 0.0, 1.1],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.6, 1.0, 1.4, 1.1, 0.0],
]

# 边的顺序由 RoadNetwork._initialize_network() 决定：按 road_network_matrix
# 逐行、从左到右扫描非零元素得到（第0行先扫完再扫第1行……）。矩阵里非零元素
# 一共32个，所以下面每一行速度快照也必须正好是32个值，顺序跟扫描出来的边
# 一一对应，改了矩阵结构（增删道路）之后必须同步改这里，否则速度会错位到
# 别的边上。
#
# 一行表示一个速度快照，单位 km/h。
road_network_speed_matrix = [
    [45.0, 35.0, 45.0, 45.0, 35.0, 45.0, 50.0, 35.0, 50.0, 35.0, 35.0, 40.0, 30.0, 35.0, 40.0, 40.0,
     30.0, 45.0, 35.0, 40.0, 40.0, 30.0, 35.0, 40.0, 30.0, 30.0, 30.0, 35.0, 45.0, 30.0, 30.0, 35.0],
]

# 每个元素对应 road_network_speed_matrix 中同一行速度数据的生效时间，单位：分钟。
# 现在只有一个快照、从0分钟（仿真起点）就生效，也就是全程速度恒定不变。
road_network_speed_timetable = [0.0]
