import networkx as nx


def check_node_connectivity(
    graph,                  # networkx.Graph or networkx.DiGraph
    source_node,            # Source 节点 ID
    target_node,            # 目标节点 ID
    directed=True,          # 检查从source 到 target
    return_path=False,      # 是否返回两点之间最短路径
):
    """
    -------
    result : dict
        Connectivity result.

        {
            "source_node": source_node,
            "target_node": target_node,
            "source_exists": bool,
            "target_exists": bool,
            "connected": bool,
            "directed": bool,
            "path": list or None,
            "path_length_edges": int or None
        }
    """

    source_exists = source_node in graph
    target_exists = target_node in graph

    result = {
        "source_node": source_node,
        "target_node": target_node,
        "source_exists": source_exists,
        "target_exists": target_exists,
        "connected": False,
        "directed": directed,
        "path": None,
        "path_length_edges": None,
    }

    if not source_exists or not target_exists:
        return result

    if directed:
        working_graph = graph
    else:
        working_graph = graph.to_undirected()

    connected = nx.has_path(
        working_graph,
        source_node,
        target_node,
    )

    result["connected"] = connected

    if connected and return_path:
        path = nx.shortest_path(
            working_graph,
            source_node,
            target_node,
        )

        result["path"] = path
        result["path_length_edges"] = len(path) - 1

    return result