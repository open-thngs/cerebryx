from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import plotly.graph_objects as go
from nicegui import ui as ng

# --- Cluster placement ---
_CLUSTER_PLACEMENT_ATTEMPTS = 300
_CLUSTER_EDGE_MARGIN = 1.2        # clusters kept this * spread inside brain_radius
_MIN_SEP_SCALE = 0.8              # min separation = max(this * spread, _MIN_SEP_ABS)
_MIN_SEP_ABS = 1.5
_BETA_A = 2.2                     # Beta shape a for brain-like inner-region bias
_BETA_B = 4.8                     # Beta shape b

# --- Neuron placement ---
_NEURON_PLACEMENT_ATTEMPTS = 60
_NEURON_SCALE = 0.48              # normal spread within cluster (fraction of radius)
_NEURON_MAX_FRAC = 0.95           # fallback: max fraction of cluster radius
_NEURON_EDGE_FRAC = 0.08          # fallback: max fraction of brain radius at boundary

# --- Connectivity ---
_BRAIN_LIKE_SCALE = 1.35          # connection density multiplier in brain-like mode
_LONG_RANGE_ATTEMPT_FRAC = 0.06
_LONG_RANGE_SIGMA_MULT = 1.8
_SMALL_WORLD_RATE = 0.008
_SMALL_WORLD_SIGMA_MULT = 2.3

# --- Visualization ---
_SPHERE_THETA = 22
_SPHERE_PHI = 16
_CLUSTER_PALETTE = [
    "#2EC4B6", "#FF9F1C", "#E71D36", "#3A86FF", "#8338EC",
    "#06D6A0", "#F15BB5", "#8AC926", "#1982C4", "#FFCA3A",
]


@dataclass
class Cluster:
    id: int
    center: np.ndarray
    radius: float
    neuron_ids: list[int] = field(default_factory=list)


@dataclass
class Neuron:
    id: int
    cluster_id: int
    position: np.ndarray


@dataclass
class Edge:
    source: int
    target: int
    weight: float
    type: Literal["local", "long"]


def gaussian_prob(distance: float | np.ndarray, sigma: float) -> float | np.ndarray:
    safe_sigma = max(0.05, float(sigma))
    return np.exp(-(distance**2) / (2.0 * safe_sigma**2))


def random_point_in_sphere(radius: float, rng: np.random.Generator) -> np.ndarray:
    direction = rng.normal(size=3)
    norm = np.linalg.norm(direction)
    while norm == 0.0:
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
    return direction / norm * (radius * (rng.random() ** (1.0 / 3.0)))


def generate_clusters(
    num_clusters: int,
    brain_radius: float,
    cluster_spread: float,
    rng: np.random.Generator,
    brain_like_mode: bool,
) -> list[Cluster]:
    clusters: list[Cluster] = []
    min_separation = max(_MIN_SEP_SCALE * cluster_spread, _MIN_SEP_ABS)
    placement_radius = max(1.0, brain_radius - _CLUSTER_EDGE_MARGIN * cluster_spread)

    for cluster_id in range(num_clusters):
        placed = False
        for _ in range(_CLUSTER_PLACEMENT_ATTEMPTS):
            if brain_like_mode:
                r = max(1.0, placement_radius * rng.beta(_BETA_A, _BETA_B))
                center = random_point_in_sphere(r, rng)
            else:
                center = random_point_in_sphere(placement_radius, rng)

            if not clusters or min(np.linalg.norm(center - c.center) for c in clusters) >= min_separation:
                placed = True
                break

        if not placed:
            center = random_point_in_sphere(max(1.0, brain_radius - cluster_spread), rng)

        clusters.append(Cluster(id=cluster_id, center=center, radius=cluster_spread))

    return clusters


def generate_neurons(
    clusters: list[Cluster],
    neurons_per_cluster: int,
    brain_radius: float,
    rng: np.random.Generator,
) -> list[Neuron]:
    neurons: list[Neuron] = []
    neuron_id = 0

    for cluster in clusters:
        for _ in range(neurons_per_cluster):
            for _attempt in range(_NEURON_PLACEMENT_ATTEMPTS):
                point = cluster.center + rng.normal(scale=cluster.radius * _NEURON_SCALE, size=3)
                if np.linalg.norm(point) <= brain_radius:
                    break
            else:
                direction = point - cluster.center
                norm = float(np.linalg.norm(direction)) + 1e-9
                point = cluster.center + direction / norm * min(
                    cluster.radius * _NEURON_MAX_FRAC, brain_radius * _NEURON_EDGE_FRAC
                )

            neurons.append(Neuron(id=neuron_id, cluster_id=cluster.id, position=point))
            cluster.neuron_ids.append(neuron_id)
            neuron_id += 1

    return neurons


def build_connections(
    neurons: list[Neuron],
    clusters: list[Cluster],
    sigma: float,
    p_long: float,
    rng: np.random.Generator,
    brain_like_mode: bool,
) -> list[Edge]:
    edges: list[Edge] = []
    scale = _BRAIN_LIKE_SCALE if brain_like_mode else 1.0

    positions = np.array([n.position for n in neurons], dtype=float)

    for cluster in clusters:
        ids = np.array(cluster.neuron_ids, dtype=int)
        if ids.size < 2:
            continue

        local_pos = positions[ids]
        dist = np.linalg.norm(local_pos[:, None, :] - local_pos[None, :, :], axis=2)
        prob = gaussian_prob(dist, sigma) * scale
        np.fill_diagonal(prob, 0.0)
        upper_i, upper_j = np.triu_indices(ids.size, k=1)
        accepted = rng.random(size=upper_i.size) < np.clip(prob[upper_i, upper_j], 0.0, 1.0)

        for idx in np.where(accepted)[0]:
            i, j = ids[upper_i[idx]], ids[upper_j[idx]]
            weight = float(np.clip(prob[upper_i[idx], upper_j[idx]], 0.05, 1.0))
            edges.append(Edge(source=int(i), target=int(j), weight=weight, type="local"))

    p_long_eff = float(np.clip(p_long * scale, 0.0, 1.0))
    for left in range(len(clusters)):
        for right in range(left + 1, len(clusters)):
            left_ids = clusters[left].neuron_ids
            right_ids = clusters[right].neuron_ids
            if not left_ids or not right_ids:
                continue

            attempts = max(1, int(np.ceil((len(left_ids) + len(right_ids)) * p_long_eff * _LONG_RANGE_ATTEMPT_FRAC)))
            for _ in range(attempts):
                if rng.random() > p_long_eff:
                    continue
                i = int(rng.choice(left_ids))
                j = int(rng.choice(right_ids))
                d = float(np.linalg.norm(positions[i] - positions[j]))
                weight = float(np.clip(0.7 * gaussian_prob(d, _LONG_RANGE_SIGMA_MULT * sigma), 0.03, 0.8))
                edges.append(Edge(source=i, target=j, weight=weight, type="long"))

    if brain_like_mode and len(neurons) > 3:
        for _ in range(max(1, int(_SMALL_WORLD_RATE * len(neurons)))):
            i = int(rng.integers(0, len(neurons)))
            j = int(rng.integers(0, len(neurons)))
            if i == j or neurons[i].cluster_id == neurons[j].cluster_id:
                continue
            d = float(np.linalg.norm(positions[i] - positions[j]))
            weight = float(np.clip(0.55 * gaussian_prob(d, _SMALL_WORLD_SIGMA_MULT * sigma), 0.02, 0.65))
            edges.append(Edge(source=i, target=j, weight=weight, type="long"))

    return edges


def make_sphere_surface(center: np.ndarray, radius: float, color: str) -> go.Surface:
    theta = np.linspace(0, 2 * np.pi, _SPHERE_THETA)
    phi = np.linspace(0, np.pi, _SPHERE_PHI)
    x = center[0] + radius * np.outer(np.cos(theta), np.sin(phi))
    y = center[1] + radius * np.outer(np.sin(theta), np.sin(phi))
    z = center[2] + radius * np.outer(np.ones_like(theta), np.cos(phi))
    return go.Surface(
        x=x, y=y, z=z,
        opacity=0.1,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        hoverinfo="skip",
        name="cluster volume",
    )


def visualize(
    neurons: list[Neuron],
    clusters: list[Cluster],
    edges: list[Edge],
    brain_radius: float,
    max_edges_render: int = 2600,
) -> go.Figure:
    fig = go.Figure()
    positions = np.array([n.position for n in neurons], dtype=float)
    cluster_ids = np.array([n.cluster_id for n in neurons], dtype=int)

    for cluster in clusters:
        color = _CLUSTER_PALETTE[cluster.id % len(_CLUSTER_PALETTE)]
        fig.add_trace(make_sphere_surface(cluster.center, cluster.radius, color))

    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode="markers",
            marker={"size": 2.6, "color": cluster_ids, "colorscale": "Turbo", "opacity": 0.95},
            name="neurons",
            hovertemplate="Neuron %{pointNumber}<extra></extra>",
        )
    )

    local_x: list[float | None] = []
    local_y: list[float | None] = []
    local_z: list[float | None] = []
    long_x: list[float | None] = []
    long_y: list[float | None] = []
    long_z: list[float | None] = []

    for edge in edges[:max_edges_render]:
        p1, p2 = positions[edge.source], positions[edge.target]
        if edge.type == "local":
            local_x += [float(p1[0]), float(p2[0]), None]
            local_y += [float(p1[1]), float(p2[1]), None]
            local_z += [float(p1[2]), float(p2[2]), None]
        else:
            long_x += [float(p1[0]), float(p2[0]), None]
            long_y += [float(p1[1]), float(p2[1]), None]
            long_z += [float(p1[2]), float(p2[2]), None]

    if local_x:
        fig.add_trace(go.Scatter3d(
            x=local_x, y=local_y, z=local_z,
            mode="lines",
            line={"width": 1.1, "color": "rgba(35, 142, 107, 0.42)"},
            hoverinfo="skip",
            name="local",
        ))

    if long_x:
        fig.add_trace(go.Scatter3d(
            x=long_x, y=long_y, z=long_z,
            mode="lines",
            line={"width": 1.5, "color": "rgba(245, 77, 77, 0.5)"},
            hoverinfo="skip",
            name="long-range",
        ))

    axis_cfg = {"visible": False, "range": [-brain_radius, brain_radius], "showgrid": False, "zeroline": False}
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        scene={
            "xaxis": axis_cfg,
            "yaxis": axis_cfg,
            "zaxis": axis_cfg,
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
        formatter: Callable[[float], str],
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
                "Long-range connection probability", 0.0, 0.2, 0.03, 0.005, lambda v: f"{float(v):.3f}"
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
            int(num_clusters.value), float(brain_radius.value), float(cluster_spread.value),
            rng, bool(brain_like_mode.value),
        )
        neurons = generate_neurons(clusters, int(neurons_per_cluster.value), float(brain_radius.value), rng)
        edges = build_connections(
            neurons, clusters,
            sigma=float(sigma.value), p_long=float(long_prob.value),
            rng=rng, brain_like_mode=bool(brain_like_mode.value),
        )
        fig = visualize(neurons, clusters, edges, float(brain_radius.value))

        plot_holder.clear()
        with plot_holder:
            ng.plotly(fig).style("width: 100%; height: 80vh;")

        local_edges = sum(1 for e in edges if e.type == "local")
        stats.text = (
            f"Generated {len(clusters)} clusters, {len(neurons)} neurons, "
            f"{local_edges} local edges, {len(edges) - local_edges} long-range edges."
        )

    with container:
        regenerate()


ui()
ng.run(title="Cerebryx Brain Topology Simulator")
