import osmnx as ox
import networkx as nx
from pathlib import Path


GRAPHML_PATH = Path(
    "processed_network/step_02_with_ids/network_with_ids.graphml"
)


def get_component_length(G, component_nodes):
    subgraph = G.subgraph(component_nodes)

    total_length = 0.0

    for _, _, _, data in subgraph.edges(keys=True, data=True):
        total_length += float(data.get("length", 0))

    return total_length


def inspect_weak_components(graphml_path=GRAPHML_PATH):
    G = ox.load_graphml(graphml_path)

    components = list(nx.weakly_connected_components(G))

    component_records = []

    for index, component_nodes in enumerate(components, start=1):
        subgraph = G.subgraph(component_nodes)

        node_count = subgraph.number_of_nodes()
        edge_count = subgraph.number_of_edges()
        total_length = get_component_length(G, component_nodes)

        xs = [
            float(G.nodes[node]["x"])
            for node in component_nodes
            if "x" in G.nodes[node]
        ]

        ys = [
            float(G.nodes[node]["y"])
            for node in component_nodes
            if "y" in G.nodes[node]
        ]

        component_records.append(
            {
                "component_id": f"component_{index:04d}",
                "node_count": node_count,
                "edge_count": edge_count,
                "total_length_m": total_length,
                "centroid_x": sum(xs) / len(xs) if xs else None,
                "centroid_y": sum(ys) / len(ys) if ys else None,
                "min_x": min(xs) if xs else None,
                "max_x": max(xs) if xs else None,
                "min_y": min(ys) if ys else None,
                "max_y": max(ys) if ys else None,
            }
        )

    component_records = sorted(
        component_records,
        key=lambda x: x["node_count"],
        reverse=True,
    )

    print("=" * 60)
    print("WEAKLY CONNECTED COMPONENTS")
    print("=" * 60)
    print(f"Number of weak components: {len(component_records)}")
    print()

    for record in component_records[:20]:
        print(
            record["component_id"],
            "nodes:",
            record["node_count"],
            "edges:",
            record["edge_count"],
            "length_m:",
            round(record["total_length_m"], 2),
            "centroid:",
            f"({record['centroid_x']:.6f}, {record['centroid_y']:.6f})",
        )

    return component_records

if __name__ == "__main__":
    inspect_weak_components()