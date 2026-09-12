"""基础仿真测试使用的默认充电站配置。

default_stations_config 可以直接作为 EVChargingEnv 的 stations_config 构造
参数传入——stations_config 是 StationManager 内部会逐项转成 ChargingStation
对象的一份"JSON式"配置列表，这里每个字典的 key 名字要跟
ChargingStation.__init__() 的形参一一对应（见 sim_env/base_station.py）：

    from sim_env.default_stations import default_stations_config
    env = EVChargingEnv(
        ...,
        stations_config=default_stations_config,
        ...,
    )

对应 default_road_network.py 的10节点路网：两个站点分别挂在 node5 / node8 上
（跟旧版本一致）。

⚠️ num_chargers / power_kw / mean_service_time_minutes / service_time_std_minutes
   现在都是按 [slow, fast] 两个充电桩类型给的列表（ChargingStation 目前区分这
   两种类型），旧版本只有 fast 一种、没有 slow 数据，下面 slow 那一半是新补的
   占位示例值。
⚠️ arrival_rate_per_hour 是 [slow到达率数组, fast到达率数组]，每个数组长度必须
   是24（ChargingStation 内部按"整点小时"查表，下标i对应"第i点整"这个时段）。
   下面用 _build_arrival_rate_per_hour() 生成的早晚通勤/夜间慢充两条曲线只是
   示例假设，不是真实调研数据——接入真实场景数据时应该整体替换掉这个文件的
   内容，只要 stations_config 的字段名不变，调用方（EVChargingEnv）不需要
   跟着改。
⚠️ 到达率的具体数值不是随便给的：话务强度 ρ_norm = λ·(μ/60)/k 必须明显小于1
   （k=充电桩数，μ=平均服务时间分钟数），否则 waiting_time_minutes 用的
   Allen-Cunneen近似公式会在ρ接近/超过k时发散、算出离谱的天文数字。下面每组
   数值都是先定好"峰值话务强度≈0.6~0.67、平时≈0.1~0.13"这个目标区间，再反推
   出的到达率，不是拍脑袋的整数。
"""

from typing import Any


def _build_arrival_rate_per_hour(peak_hours: list[int], peak_rate: float, base_rate: float) -> list[float]:
    """生成一条24小时到达率曲线(辆/小时)：peak_hours这些整点小时取peak_rate，
    其余整点小时取base_rate。长度固定是24，对应ChargingStation内部按整点小时
    查表的_ARRIVAL_RATE_TIMETABLE_MINUTES。
    """
    return [peak_rate if hour in peak_hours else base_rate for hour in range(24)]


# fast充电桩：主要是路上临时补电，到达高峰跟通勤时间重合（早7-9点、晚17-19点）。
_FAST_PEAK_HOURS = [7, 8, 9, 17, 18, 19]
# slow充电桩：主要是夜间长时间停车慢充，到达高峰在夜间（22点-次日5点）。
_SLOW_PEAK_HOURS = [22, 23, 0, 1, 2, 3, 4, 5]


default_stations_config: list[dict[str, Any]] = [
    {
        "station_id": "station_001",
        "mapped_node": 5,
        "access_distance_km": 0.0,
        # 以下四项都是 [slow, fast]
        "num_chargers": [3, 2],
        "power_kw": [7.0, 60.0],
        "mean_service_time_minutes": [240.0, 30.0],  # slow充满约4小时；fast快充约30分钟
        "service_time_std_minutes": [60.0, 8.0],
        "arrival_rate_per_hour": [
            _build_arrival_rate_per_hour(_SLOW_PEAK_HOURS, peak_rate=0.5, base_rate=0.1),  # slow: 峰值ρ≈0.67，平时≈0.13
            _build_arrival_rate_per_hour(_FAST_PEAK_HOURS, peak_rate=2.5, base_rate=0.5),  # fast: 峰值ρ≈0.63，平时≈0.13
        ],
    },
    {
        "station_id": "station_002",
        "mapped_node": 8,
        "access_distance_km": 0.0,
        "num_chargers": [2, 1],
        "power_kw": [7.0, 90.0],
        "mean_service_time_minutes": [240.0, 20.0],  # fast桩功率更高(90kW)，服务时间比station_001短
        "service_time_std_minutes": [60.0, 5.0],
        "arrival_rate_per_hour": [
            _build_arrival_rate_per_hour(_SLOW_PEAK_HOURS, peak_rate=0.3, base_rate=0.06),  # slow: 峰值ρ≈0.6，平时≈0.12
            _build_arrival_rate_per_hour(_FAST_PEAK_HOURS, peak_rate=2.0, base_rate=0.4),  # fast: 只有1个桩，峰值ρ≈0.67，平时≈0.13
        ],
    },
]
