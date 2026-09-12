"""算法模板：约定所有路径规划算法和环境交互的统一方式。"""


class RoutingAlgorithm:
    """所有算法的公共基类。

    interaction() 是驱动循环，对所有算法都是同一套流程，已经在基类里写好，
    不需要每个算法重复实现；decide_path() 是每个算法自己的决策逻辑，子类
    必须实现，不实现就报错——基类本身不提供任何默认的决策逻辑。
    """

    def decide_path(self, env):
        """给定当前环境状态(env)，返回下一段要走的路径（节点id组成的列表）。
        子类必须实现这个函数。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 必须实现 decide_path()，基类不提供默认决策逻辑"
        )

    def interaction(self, env) -> dict:
        """驱动循环：不断调用 env.step() 推进仿真，只有在"当前没有方案在跑"
        （env.has_active_plan() 为 False）时，才调用 decide_path() 决定下一段
        路径；对所有算法都是同一套流程，区别只在于 decide_path() 一次给多长
        的路径——一次给出完整路径就是一次性算法，一次只给一条边就是每到一个
        节点都重新决策一次。

        车辆到达终点（当前节点等于目的地节点）时循环结束，返回最后一次的
        环境状态。
        """
        vehicle = env.get_vehicle()  # 车辆对象，用来查当前位置和目的地
        destination_node_id = vehicle.get_demand_info()["destination_node_id"]  # 这次行程的终点，出行过程中不变

        state = env.get_state()  # 当前的环境状态，每次step()之后会更新成最新的
        current_node_id = vehicle.get_dynamic_state()["current_node_id"]  # 车辆当前所在节点

        while current_node_id != destination_node_id:
            if not env.has_active_plan():
                next_path = self.decide_path(env)  # 没有方案在跑，问算法要下一段路径
                state = env.step(action={"path": next_path})
            else:
                state = env.step(action=None)  # 有方案在跑，接着走一条边就行，不用重新决策

            current_node_id = vehicle.get_dynamic_state()["current_node_id"]

        return state
