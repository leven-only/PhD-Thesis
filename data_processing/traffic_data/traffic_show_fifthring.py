def process_lonlat_and_speed(
    file_path="graph_info.xlsx",
    hour_index=0,
    node_num=44
):
    import re
    import numpy as np
    import pandas as pd

    lon_lat_df = pd.read_excel(file_path, sheet_name="lon_lat", header=None)
    speed_df = pd.read_excel(file_path, sheet_name="processed_speed", header=None)

    def parse_edge_coord(value):
        import pandas as pd

        if pd.isna(value):
            return None

        value = str(value).strip()

        if value in ["0", "0.0", ""]:
            return None

        try:
            start_str, end_str = value.split("-")

            lon1, lat1 = start_str.split(",")
            lon2, lat2 = end_str.split(",")

            lon1 = float(lon1)
            lat1 = float(lat1)
            lon2 = float(lon2)
            lat2 = float(lat2)

            return (lon1, lat1), (lon2, lat2)

        except Exception:
            return None

    node_coords = {i: [] for i in range(node_num)}
    directed_edges = []

    speed_values = speed_df.iloc[hour_index].values.astype(float)
    speed_idx = 0

    for i in range(lon_lat_df.shape[0]):
        for j in range(lon_lat_df.shape[1]):

            parsed = parse_edge_coord(lon_lat_df.iloc[i, j])

            if parsed is None:
                continue

            start_coord, end_coord = parsed

            node_coords[i].append(start_coord)
            node_coords[j].append(end_coord)

            speed = speed_values[speed_idx]
            directed_edges.append((i, j, speed))

            speed_idx += 1

    node_pos = {}

    for node_id, coords in node_coords.items():
        if len(coords) == 0:
            continue

        coords = np.array(coords)
        node_pos[node_id] = (
            coords[:, 0].mean(),
            coords[:, 1].mean()
        )

    edge_speed_dict = {}

    for i, j, speed in directed_edges:
        edge_key = tuple(sorted((i, j)))

        if edge_key not in edge_speed_dict:
            edge_speed_dict[edge_key] = []

        edge_speed_dict[edge_key].append(speed)

    edges = []

    for (i, j), speeds in edge_speed_dict.items():
        avg_speed = float(np.mean(speeds))
        edges.append((i, j, avg_speed))

    return node_pos, edges

def draw_traffic_show(
    node_pos,
    edges,
    output_path="traffic_show.jpg",
    dpi=600,
    line_width=2.0,
    node_size=35,
    cmap_name="viridis",
    show=True
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    line_segments = []
    speeds = []

    for u, v, speed in edges:
        if u not in node_pos or v not in node_pos:
            continue

        x1, y1 = node_pos[u]
        x2, y2 = node_pos[v]

        line_segments.append([(x1, y1), (x2, y2)])
        speeds.append(speed)

    speeds = np.array(speeds)

    if len(line_segments) == 0:
        raise ValueError("没有可绘制的路段。")

    norm = plt.Normalize(vmin=speeds.min(), vmax=speeds.max())

    lc = LineCollection(
        line_segments,
        cmap=cmap_name,
        norm=norm,
        linewidths=line_width
    )

    lc.set_array(speeds)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.add_collection(lc)

    xs = [node_pos[i][0] for i in node_pos]
    ys = [node_pos[i][1] for i in node_pos]

    ax.scatter(
        xs,
        ys,
        s=node_size,
        c="black",
        zorder=3
    )

    margin_x = (max(xs) - min(xs)) * 0.03
    margin_y = (max(ys) - min(ys)) * 0.03

    ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
    ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    cbar = plt.colorbar(lc, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Speed")

    plt.savefig(
        output_path,
        dpi=dpi,
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.02
    )

    if show:
        plt.show()

    plt.close(fig)

    print(f"图像已保存: {output_path}")

if __name__ == '__main__':
    node_pos, edges = process_lonlat_and_speed(
        file_path="graph_info.xlsx",
        hour_index=7,
        node_num=44
    )

    draw_traffic_show(
        node_pos=node_pos,
        edges=edges,
        output_path="traffic_hour_0.jpg"
    )