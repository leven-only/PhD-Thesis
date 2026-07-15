# RoadNetwork 使用说明

`RoadNetwork` 用于从邻接矩阵、速度数据和速度时间表创建一个动态路网。内部路网使用 `networkx.DiGraph` 保存，矩阵的行列下标直接作为节点 ID。

## 1. 依赖

使用前需要安装 NetworkX：

```bash
pip install networkx
```

导入类：

```python
from sim_env.core_road_network import RoadNetwork
```

## 2. 创建路网对象

```python
road_network = RoadNetwork(
    matrix=road_matrix,
    speed_matrix=speed_matrix,
    speed_timetable=speed_timetable,
    start_time=0.0,
)
```

构造函数格式：

```python
RoadNetwork(
    matrix: Any,
    speed_matrix: Any,
    speed_timetable: Any,
    start_time: float,
)
```

### 2.1 `matrix`

路网邻接矩阵，必须是非空方阵。

- 行下标表示道路起点。
- 列下标表示道路终点。
- `0` 表示两个节点之间没有道路。
- 非零值表示道路长度，单位为 km。
- 对角线必须全部为 `0`。
- 不能包含负数。

例如：

```python
road_matrix = [
    [0.0, 2.0, 4.0],
    [0.0, 0.0, 3.0],
    [1.0, 0.0, 0.0],
]
```

该矩阵创建以下节点和有向道路：

```text
节点：0、1、2

道路：
0 -> 1，长度 2.0 km
0 -> 2，长度 4.0 km
1 -> 2，长度 3.0 km
2 -> 0，长度 1.0 km
```

`RoadNetwork` 使用 `nx.DiGraph`，因此 `0 -> 1` 和 `1 -> 0` 是两条不同的道路。如果道路可以双向通行，邻接矩阵中两个方向都必须为非零值。

`matrix` 可以传入以下二维数据：

- Python 二维列表；
- NumPy 二维数组；
- pandas `DataFrame`。

### 2.2 `speed_matrix`

速度快照数据，每一行表示一个时刻的全部道路速度，单位为 km/h。

```python
speed_matrix = [
    [10.0, 20.0, 30.0, 40.0],
    [11.0, 21.0, 31.0, 41.0],
]
```

每行速度值的数量必须等于 `matrix` 中非零元素的数量。速度不能为负数。

道路和速度列按照邻接矩阵从上到下、每行从左到右的顺序对应。以上示例的对应关系为：

| 速度列下标 | 矩阵位置 | 道路 |
|---:|---:|---:|
| 0 | `matrix[0][1]` | `0 -> 1` |
| 1 | `matrix[0][2]` | `0 -> 2` |
| 2 | `matrix[1][2]` | `1 -> 2` |
| 3 | `matrix[2][0]` | `2 -> 0` |

因此第一行速度数据表示：

```text
0 -> 1：10 km/h
0 -> 2：20 km/h
1 -> 2：30 km/h
2 -> 0：40 km/h
```

`speed_matrix` 可以传入 Python 二维列表、NumPy 二维数组或 pandas `DataFrame`。

### 2.3 `speed_timetable`

速度时间表。每个时间对应 `speed_matrix` 中相同下标的一行速度数据。

```python
speed_timetable = [0.0, 60.0]
```

对应关系为：

```text
时间 0.0  -> speed_matrix[0]
时间 60.0 -> speed_matrix[1]
```

要求：

- 时间数量必须等于 `speed_matrix` 的行数；
- 时间必须严格递增；
- 可以传入一维列表或单列二维数据。

### 2.4 `start_time`

环境开始时间，数值类型。创建路网和调用 `reset()` 时，都会使用该时间对应的速度快照。

```python
start_time = 30.0
```

速度选择采用向下取值：选择时间表中不大于当前时间的最大时间。

例如：

```python
speed_timetable = [0.0, 60.0, 120.0]
```

| 当前时间 | 使用的速度行 |
|---:|---:|
| `30.0` | `speed_matrix[0]` |
| `60.0` | `speed_matrix[1]` |
| `100.0` | `speed_matrix[1]` |
| `120.0` | `speed_matrix[2]` |
| `500.0` | `speed_matrix[2]` |

如果时间早于时间表的第一个时间，则使用第一行速度；如果时间超过最大时间，则使用最后一行速度。

## 3. 完整创建示例

```python
from sim_env.core_road_network import RoadNetwork

road_matrix = [
    [0.0, 2.0, 4.0],
    [0.0, 0.0, 3.0],
    [1.0, 0.0, 0.0],
]

speed_matrix = [
    [10.0, 20.0, 30.0, 40.0],
    [11.0, 21.0, 31.0, 41.0],
]

speed_timetable = [0.0, 60.0]

road_network = RoadNetwork(
    matrix=road_matrix,
    speed_matrix=speed_matrix,
    speed_timetable=speed_timetable,
    start_time=0.0,
)
```

创建完成后，可以直接访问内部的 NetworkX 图对象：

```python
graph = road_network.graph

print(list(graph.nodes))
# [0, 1, 2]

print(list(graph.edges))
# [(0, 1), (0, 2), (1, 2), (2, 0)]
```

每条边包含以下属性：

```python
print(graph.edges[0, 1])
```

```python
{
    "edge_id": "edge_0000",
    "length_km": 2.0,
    "speed_kph": 10.0,
    "travel_time_seconds": 720.0,
}
```

## 4. 对外函数

### 4.1 `reset()`

```python
road_network.reset()
```

输入：无。

输出：`None`。

作用：将所有道路速度恢复到 `start_time` 对应的速度快照。路网节点和道路拓扑不会重新创建。

### 4.2 `step(current_time)`

```python
road_network.step(current_time=60.0)
```

函数格式：

```python
step(current_time: float) -> None
```

输入：

- `current_time`：当前仿真时间。

输出：`None`。

作用：根据 `current_time` 在 `speed_timetable` 中向下查找对应下标，然后用 `speed_matrix` 中相同下标的速度行更新所有道路。

如果当前道路已经使用目标速度快照，则不会重复更新。

### 4.3 `get_state()`

```python
state = road_network.get_state()
```

函数格式：

```python
get_state() -> dict[str, Any]
```

输入：无。

输出格式：

```python
{
    "nodes": [0, 1, 2],
    "node_count": 3,
    "edge_count": 4,
    "current_speed_snapshot_index": 0,
    "current_speed_time": 0.0,
    "edges": [
        {
            "node_u": 0,
            "node_v": 1,
            "edge_id": "edge_0000",
            "length_km": 2.0,
            "speed_kph": 10.0,
            "travel_time_seconds": 720.0,
        },
    ],
}
```

字段说明：

| 字段 | 类型 | 含义 |
|---|---|---|
| `nodes` | `list[int]` | 图中的节点列表 |
| `node_count` | `int` | 节点数量 |
| `edge_count` | `int` | 有向道路数量 |
| `current_speed_snapshot_index` | `int` | 当前速度快照的行下标 |
| `current_speed_time` | `float` | 当前速度快照对应的时间 |
| `edges` | `list[dict]` | 当前全部道路及其属性 |

### 4.4 `get_edge_between(node_u, node_v)`

```python
edge = road_network.get_edge_between(0, 1)
```

函数格式：

```python
get_edge_between(
    node_u: int,
    node_v: int,
) -> dict[str, Any]
```

输入：

- `node_u`：道路起点，即邻接矩阵的行下标；
- `node_v`：道路终点，即邻接矩阵的列下标。

输出：NetworkX 中对应道路的属性字典。

```python
{
    "edge_id": "edge_0000",
    "length_km": 2.0,
    "speed_kph": 10.0,
    "travel_time_seconds": 720.0,
}
```

如果道路不存在，会抛出：

```text
ValueError: 节点之间不存在道路: 1 -> 0
```

返回的是图中边属性的直接引用。修改该字典会直接修改图中的道路属性，一般建议只读取，由 `step()` 统一更新速度。

## 5. 常用查询示例

查询节点和道路数量：

```python
print(road_network.graph.number_of_nodes())
print(road_network.graph.number_of_edges())
```

查询两个节点之间是否有道路：

```python
has_edge = road_network.graph.has_edge(0, 1)
```

查询某个节点可以到达的相邻节点：

```python
neighbors = list(road_network.graph.successors(0))
```

读取道路当前速度：

```python
speed = road_network.get_edge_between(0, 1)["speed_kph"]
```

读取道路当前通行时间：

```python
travel_time = road_network.get_edge_between(0, 1)[
    "travel_time_seconds"
]
```

## 6. 输入数据检查

创建对象时会自动检查：

- `matrix` 是否为非空方阵；
- `matrix` 是否包含负数；
- `matrix` 对角线是否全部为 `0`；
- `matrix` 是否至少包含一条道路；
- `speed_matrix` 是否为空；
- 每行速度数量是否等于邻接矩阵非零元素数量；
- 速度是否包含负数；
- 时间表元素数量是否等于速度数据行数；
- 时间表是否严格递增。

输入不符合要求时会抛出 `TypeError` 或 `ValueError`。
