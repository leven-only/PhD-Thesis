from pathlib import Path

import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent

GRAPHML_PATH = (
    BASE_DIR
    / "processed_network"
    / "step_04_largest_component"
    / "network_largest_component.graphml"
)

DEAD_END_CSV_PATH = (
    BASE_DIR
    / "processed_network"
    / "step_05_dead_end_diagnosis"
    / "dead_end_nodes.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "processed_network"
    / "step_05_dead_end_diagnosis"
    / "dead_end_candidates_map.jpg"
)


def draw_dead_end_nodes_on_network(
    graphml_path=GRAPHML_PATH,
    dead_end_csv_path=DEAD_END_CSV_PATH,
    output_path=OUTPUT_PATH,
    node_id_column="node",
    dead_end_type_column=None,
    selected_types=None,
    dpi=600,
):
    graphml_path = Path(graphml_path)
    dead_end_csv_path = Path(dead_end_csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    G = ox.load_graphml(graphml_path)
    dead_ends = pd.read_csv(dead_end_csv_path)

    if node_id_column not in dead_ends.columns:
        raise ValueError(f"Column not found: {node_id_column}")

    if dead_end_type_column and selected_types:
        if dead_end_type_column not in dead_ends.columns:
            raise ValueError(f"Column not found: {dead_end_type_column}")

        dead_ends = dead_ends[
            dead_ends[dead_end_type_column].isin(selected_types)
        ]

    candidate_nodes = set(dead_ends[node_id_column].astype(str))

    node_colors = []
    node_sizes = []

    for node in G.nodes:
        if str(node) in candidate_nodes:
            node_colors.append("red")
            node_sizes.append(18)
        else:
            node_colors.append("none")
            node_sizes.append(0)

    fig, ax = ox.plot_graph(
        G,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color="lightgray",
        edge_linewidth=0.5,
        bgcolor="white",
        show=False,
        close=False,
    )

    ax.set_axis_off()

    plt.savefig(
        output_path,
        dpi=dpi,
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

    print(f"Drawn nodes: {len(candidate_nodes)}")
    print(f"Saved to: {output_path}")

    return candidate_nodes

if __name__ == '__main__':
    draw_dead_end_nodes_on_network()