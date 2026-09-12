"""DijkstraRouting：基于networkx最短路算法的一次性路径规划算法。"""

import networkx as nx

from _03_algorithms.base_algorithm import RoutingAlgorithm


class DijkstraRouting(RoutingAlgorithm):
    """经典一次性Dijkstra：决策时算出一整条从当前位置到终点的最短路径，
    一次性交给环境执行，中途不再重新规划。
    """

    def __init__(self, weight: str = "length_km") -> None:
        self.weight = weight  # 最短路用哪个边属性当权重：length_km(距离)或travel_time_seconds(时间)

    def decide_path(self, env) -> list:
        """在路网图上，从车辆当前节点到目的地算一次最短路，返回完整节点路径。"""
        vehicle = env.get_vehicle()
        current_node_id = vehicle.get_dynamic_state()["current_node_id"]  # 车辆当前所在节点，路径从这里出发
        destination_node_id = vehicle.get_demand_info()["destination_node_id"]  # 这次行程的终点

        graph = env.get_road_network().graph  # RoadNetwork内部已经是networkx图，直接拿来用，不用重新组装
        return nx.dijkstra_path(graph, current_node_id, destination_node_id, weight=self.weight)
