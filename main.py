from typing import Any

import numpy as np
import plotly.graph_objects as go
from nicegui import ui as ng


def gaussian_prob(distance: np.ndarray, sigma: float) -> np.ndarray:
    safe_sigma = max(0.05, float(sigma))
    return np.exp(-(distance**2) / (2.0 * safe_sigma**2))


def random_point_in_sphere(radius: float, rng: np.random.Generator) -> np.ndarray:
    direction = rng.normal(size=3)
    norm = np.linalg.norm(direction)
    while norm == 0.0:
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
    direction = direction / norm
    distance = radius * (rng.random() ** (1.0 / 3.0))
    return direction * distance


def generate_clusters(
    num_clusters: int,
    brain_radius: float,
    cluster_spread: float,
    rng: np.random.Generator,
    brain_like_mode: bool,
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    max_attempts = 300
    min_separation = max(0.8 * cluster_spread, 1.5)

    for cluster_id in range(num_clusters):
        accepted = False
        for _ in range(max_attempts):
            if brain_like_mode:
                # Bias toward inner regions for a denser core structure.
                distance_factor = rng.beta(2.2, 4.8)
                center_radius = max(1.0, (brain_radius - 1.2 * cluster_spread) * distance_factor)
                center = random_point_in_sphere(center_radius, rng)
            else:
                center = random_point_in_sphere(max(1.0, brain_radius - 1.2 * cluster_spread), rng)

            if not clusters:
                accepted = True
            else:
                nearest = float("inf")
                for existing in clusters:
                    dist = float(np.linalg.norm(center - existing["center"]))
                    if dist < nearest:
                        nearest = dist
                accepted = nearest >= float(min_separation)

            if accepted:
                clusters.append(
                    {
                        "id": cluster_id,
                        "center": center,
                        "radius": cluster_spread,
                        "neuron_ids": [],
                    }
                )
                break

        if not accepted:
            # Fallback keeps generation robust at high densities.
            center = random_point_in_sphere(max(1.0, brain_radius - cluster_spread), rng)
            clusters.append(
                {
                    "id": cluster_id,
                    "center": center,
                    "radius": cluster_spread,
                    "neuron_ids": [],
                }
            )

    return clusters


def generate_neurons(
    clusters: list[dict[str, Any]],
    neurons_per_cluster: int,
    brain_radius: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    neurons: list[dict[str, Any]] = []
    neuron_id = 0

    for cluster in clusters:
        center = cluster["center"]
        radius = cluster["radius"]

        for _ in range(neurons_per_cluster):
            for _attempt in range(60):
                point = center + rng.normal(scale=radius * 0.48, size=3)
                if np.linalg.norm(point) <= brain_radius:
                    break
            else:
                direction = point - center
                norm = np.linalg.norm(direction) + 1e-9
                point = center + direction / norm * min(radius * 0.95, brain_radius * 0.08)

            neurons.append(
                {
                    "id": neuron_id,
                    "cluster_id": cluster["id"],
                    "position": point,
                }
            )
            cluster["neuron_ids"].append(neuron_id)
            neuron_id += 1

    return neurons


def build_connections(
    neurons: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    sigma: float,
    p_long: float,
    rng: np.random.Generator,
    brain_like_mode: bool,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    local_scale = 1.35 if brain_like_mode else 1.0
    long_scale = 1.35 if brain_like_mode else 1.0

    positions = np.array([n["position"] for n in neurons], dtype=float)

    for cluster in clusters:
        ids = np.array(cluster["neuron_ids"], dtype=int)
        if ids.size < 2:
            continue

        local_positions = positions[ids]
        diff = local_positions[:, None, :] - local_positions[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        prob = gaussian_prob(dist, sigma) * local_scale
        np.fill_diagonal(prob, 0.0)
        upper_i, upper_j = np.triu_indices(ids.size, k=1)
        accepted = rng.random(size=upper_i.size) < np.clip(prob[upper_i, upper_j], 0.0, 1.0)

        for idx in np.where(accepted)[0]:
            i = ids[upper_i[idx]]
            j = ids[upper_j[idx]]
            weight = float(np.clip(prob[upper_i[idx], upper_j[idx]], 0.05, 1.0))
            edges.append({"source": int(i), "target": int(j), "weight": weight, "type": "local"})

    p_long = float(np.clip(p_long * long_scale, 0.0, 1.0))
    for left in range(len(clusters)):
        for right in range(left + 1, len(clusters)):
            left_ids = clusters[left]["neuron_ids"]
            right_ids = clusters[right]["neuron_ids"]
            if not left_ids or not right_ids:
                continue

            attempts = max(1, int(np.ceil((len(left_ids) + len(right_ids)) * p_long * 0.06)))
            for _ in range(attempts):
                if rng.random() > p_long:
                    continue
                i = int(rng.choice(left_ids))
                j = int(rng.choice(right_ids))
                d = np.linalg.norm(positions[i] - positions[j])
                weight = float(np.clip(0.7 * gaussian_prob(np.array(d), 1.8 * sigma), 0.03, 0.8))
                edges.append({"source": i, "target": j, "weight": weight, "type": "long"})

    if brain_like_mode and len(neurons) > 3:
        # Small-world augmentation: sparse rewired bridges.
        extra = max(1, int(0.008 * len(neurons)))
        for _ in range(extra):
            i = int(rng.integers(0, len(neurons)))
            j = int(rng.integers(0, len(neurons)))
            if i == j or neurons[i]["cluster_id"] == neurons[j]["cluster_id"]:
                continue
            d = np.linalg.norm(positions[i] - positions[j])
            weight = float(np.clip(0.55 * gaussian_prob(np.array(d), 2.3 * sigma), 0.02, 0.65))
            edges.append({"source": i, "target": j, "weight": weight, "type": "long"})

    return edges


def make_sphere_surface(center: np.ndarray, radius: float, color: str) -> go.Surface:
    theta = np.linspace(0, 2 * np.pi, 22)
    phi = np.linspace(0, np.pi, 16)
    x = center[0] + radius * np.outer(np.cos(theta), np.sin(phi))
    y = center[1] + radius * np.outer(np.sin(theta), np.sin(phi))
    z = center[2] + radius * np.outer(np.ones_like(theta), np.cos(phi))

    return go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.1,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        hoverinfo="skip",
        name="cluster volume",
    )


def visualize(
    neurons: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    brain_radius: float,
    max_edges_render: int = 2600,
) -> go.Figure:
    fig = go.Figure()

    palette = [
        "#2EC4B6",
        "#FF9F1C",
        "#E71D36",
        "#3A86FF",
        "#8338EC",
        "#06D6A0",
        "#F15BB5",
        "#8AC926",
        "#1982C4",
        "#FFCA3A",
    ]

    positions = np.array([n["position"] for n in neurons], dtype=float)
    cluster_ids = np.array([n["cluster_id"] for n in neurons], dtype=int)

    for cluster in clusters:
        color = palette[cluster["id"] % len(palette)]
        fig.add_trace(make_sphere_surface(cluster["center"], cluster["radius"], color))

    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            marker={
                "size": 2.6,
                "color": cluster_ids,
                "colorscale": "Turbo",
                "opacity": 0.95,
            },
            name="neurons",
            hovertemplate="Neuron %{pointNumber}<extra></extra>",
        )
    )

    render_edges = edges[:max_edges_render]
    local_x: list[float | None] = []
    local_y: list[float | None] = []
    local_z: list[float | None] = []
    long_x: list[float | None] = []
    long_y: list[float | None] = []
    long_z: list[float | None] = []

    for edge in render_edges:
        source = edge["source"]
        target = edge["target"]
        p1 = positions[source]
        p2 = positions[target]

        if edge["type"] == "local":
            local_x.extend([float(p1[0]), float(p2[0]), None])
            local_y.extend([float(p1[1]), float(p2[1]), None])
            local_z.extend([float(p1[2]), float(p2[2]), None])
        else:
            long_x.extend([float(p1[0]), float(p2[0]), None])
            long_y.extend([float(p1[1]), float(p2[1]), None])
            long_z.extend([float(p1[2]), float(p2[2]), None])

    if local_x:
        fig.add_trace(
            go.Scatter3d(
                x=local_x,
                y=local_y,
                z=local_z,
                mode="lines",
                line={"width": 1.1, "color": "rgba(35, 142, 107, 0.42)"},
                hoverinfo="skip",
                name="local",
            )
        )

    if long_x:
        fig.add_trace(
            go.Scatter3d(
                x=long_x,
                y=long_y,
                z=long_z,
                mode="lines",
                line={"width": 1.5, "color": "rgba(245, 77, 77, 0.5)"},
                hoverinfo="skip",
                name="long-range",
            )
        )

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        scene={
            "xaxis": {
                "visible": False,
                "range": [-brain_radius, brain_radius],
                "showgrid": False,
                "zeroline": False,
            },
            "yaxis": {
                "visible": False,
                "range": [-brain_radius, brain_radius],
                "showgrid": False,
                "zeroline": False,
            },
            "zaxis": {
                "visible": False,
                "range": [-brain_radius, brain_radius],
                "showgrid": False,
                "zeroline": False,
            },
            "aspectmode": "cube",
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    return fig


def ui() -> None:
    container = ng.column().classes("w-full")

    def slider_field(
        label: str,
        min_value: float,
        max_value: float,
        value: float,
        step: float,
        formatter,
    ):
        with ng.column().style("width: 520px; gap: 0.15rem;"):
            with ng.row().classes("w-full items-center justify-between"):
                ng.label(label)
                value_label = ng.label()
            slider = ng.slider(min=min_value, max=max_value, value=value, step=step).classes("w-full")
            value_label.bind_text_from(slider, "value", backward=formatter)
        return slider

    with ng.row().style("width: 100%; gap: 20px; align-items: flex-start; flex-wrap: wrap;"):
        with ng.column().style("flex: 0 0 560px; max-width: 560px; gap: 0.55rem;"):
            ng.label("Cerebryx Brain-Topology Simulator").classes("text-h5")
            ng.label("3D clustered neuromorphic network with local and long-range connectivity.").classes(
                "text-caption"
            )

            num_clusters = slider_field("Number of clusters", 2, 18, 8, 1, lambda v: f"{int(v)}")
            neurons_per_cluster = slider_field("Neurons per cluster", 20, 300, 90, 5, lambda v: f"{int(v)}")
            cluster_spread = slider_field("Cluster spread (radius)", 0.7, 6.0, 2.1, 0.1, lambda v: f"{float(v):.1f}")
            sigma = slider_field("Sigma (locality)", 0.25, 7.5, 1.8, 0.05, lambda v: f"{float(v):.2f}")
            long_prob = slider_field(
                "Long-range connection probability",
                0.0,
                0.2,
                0.03,
                0.005,
                lambda v: f"{float(v):.3f}",
            )
            brain_radius = slider_field("Brain sphere radius", 12.0, 55.0, 28.0, 1.0, lambda v: f"{float(v):.0f}")
            seed = slider_field("Random seed", 0, 9999, 42, 1, lambda v: f"{int(v)}")
            brain_like_mode = ng.switch("Brain-like mode", value=True)

            stats = ng.label("Ready")
            ng.button("Generate Brain", on_click=lambda: regenerate()).props("color=primary")

        plot_holder = ng.column().style("flex: 1 1 740px; min-height: 80vh;")

    def regenerate() -> None:
        rng = np.random.default_rng(int(seed.value))

        clusters = generate_clusters(
            int(num_clusters.value),
            float(brain_radius.value),
            float(cluster_spread.value),
            rng,
            bool(brain_like_mode.value),
        )
        neurons = generate_neurons(clusters, int(neurons_per_cluster.value), float(brain_radius.value), rng)
        edges = build_connections(
            neurons,
            clusters,
            sigma=float(sigma.value),
            p_long=float(long_prob.value),
            rng=rng,
            brain_like_mode=bool(brain_like_mode.value),
        )

        fig = visualize(neurons, clusters, edges, float(brain_radius.value))

        plot_holder.clear()
        with plot_holder:
            ng.plotly(fig).style("width: 100%; height: 80vh;")

        local_edges = sum(1 for edge in edges if edge["type"] == "local")
        long_edges = len(edges) - local_edges
        stats.text = (
            f"Generated {len(clusters)} clusters, {len(neurons)} neurons, "
            f"{local_edges} local edges, {long_edges} long-range edges."
        )

    with container:
        regenerate()


ui()
ng.run(title="Cerebryx Brain Topology Simulator")
