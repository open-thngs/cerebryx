from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import plotly.graph_objects as go
from nicegui import ui as ng

# --- Cluster placement ---
_CLUSTER_PLACEMENT_ATTEMPTS = 300
_CLUSTER_PLACEMENT_FRAC = 0.85    # area centers within this fraction of brain_radius
_BETA_A = 2.2                     # Beta shape a — brain-like inner-region density bias
_BETA_B = 4.8                     # Beta shape b

# --- Connectivity ---
_SMALL_WORLD_RATE = 0.008         # extra cross-area bridges per neuron in brain-like mode

# --- Visualization ---
_LOCAL_EDGE_OPACITY = 0.35
_CROSS_EDGE_COLOR = "rgba(245, 77, 77, 0.7)"
_CLUSTER_PALETTE = [
    "#2EC4B6", "#FF9F1C", "#E71D36", "#3A86FF", "#8338EC",
    "#06D6A0", "#F15BB5", "#8AC926", "#1982C4", "#FFCA3A",
]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _random_unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    norm = np.linalg.norm(v)
    while norm == 0.0:
        v = rng.normal(size=3)
        norm = np.linalg.norm(v)
    return v / norm


@dataclass
class Cluster:
    id: int
    center: np.ndarray
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
    type: Literal["local", "cross"]
    cluster_id: int  # owning area for local edges, -1 for cross-area


def generate_clusters(
    num_clusters: int,
    brain_radius: float,
    rng: np.random.Generator,
    brain_like_mode: bool,
) -> list[Cluster]:
    clusters: list[Cluster] = []
    placement_radius = max(1.0, brain_radius * _CLUSTER_PLACEMENT_FRAC)
    # Enforce minimum separation so no two Voronoi seeds are trivially close.
    min_separation = max(1.5, brain_radius / (2.0 * num_clusters))

    for cluster_id in range(num_clusters):
        placed = False
        for _ in range(_CLUSTER_PLACEMENT_ATTEMPTS):
            r = placement_radius * (rng.beta(_BETA_A, _BETA_B) if brain_like_mode else rng.random() ** (1.0 / 3.0))
            center = _random_unit(rng) * max(1.0, r)
            if not clusters or min(np.linalg.norm(center - c.center) for c in clusters) >= min_separation:
                placed = True
                break
        if not placed:
            center = _random_unit(rng) * placement_radius
        clusters.append(Cluster(id=cluster_id, center=center))

    return clusters


def generate_neurons(
    num_neurons: int,
    brain_radius: float,
    rng: np.random.Generator,
    brain_like_mode: bool,
) -> list[Neuron]:
    neurons: list[Neuron] = []
    for neuron_id in range(num_neurons):
        r = brain_radius * (rng.beta(_BETA_A, _BETA_B) if brain_like_mode else rng.random() ** (1.0 / 3.0))
        neurons.append(Neuron(id=neuron_id, cluster_id=0, position=_random_unit(rng) * r))
    return neurons


def assign_clusters(neurons: list[Neuron], clusters: list[Cluster]) -> None:
    """Voronoi partition: assign each neuron to its nearest cluster center."""
    for cluster in clusters:
        cluster.neuron_ids = []

    centers = np.array([c.center for c in clusters], dtype=float)
    positions = np.array([n.position for n in neurons], dtype=float)
    dists = np.linalg.norm(positions[:, None, :] - centers[None, :, :], axis=2)

    for neuron_idx, cluster_idx in enumerate(np.argmin(dists, axis=1)):
        cid = int(cluster_idx)
        neurons[neuron_idx].cluster_id = clusters[cid].id
        clusters[cid].neuron_ids.append(neuron_idx)


def build_connections(
    neurons: list[Neuron],
    clusters: list[Cluster],
    k_local: int,
    p_cross: float,
    rng: np.random.Generator,
    brain_like_mode: bool,
) -> list[Edge]:
    edges: list[Edge] = []
    positions = np.array([n.position for n in neurons], dtype=float)
    seen: set[tuple[int, int]] = set()

    # Local: connect each neuron to its k nearest neighbors within the same area.
    for cluster in clusters:
        ids = np.array(cluster.neuron_ids, dtype=int)
        if ids.size < 2:
            continue
        local_pos = positions[ids]
        dist = np.linalg.norm(local_pos[:, None, :] - local_pos[None, :, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        k = min(k_local, ids.size - 1)
        for i_local in range(ids.size):
            for j_local in np.argsort(dist[i_local])[:k]:
                i_g, j_g = int(ids[i_local]), int(ids[int(j_local)])
                key = (min(i_g, j_g), max(i_g, j_g))
                if key in seen:
                    continue
                seen.add(key)
                d = float(dist[i_local, int(j_local)])
                weight = float(np.clip(1.0 / (1.0 + d * 0.3), 0.05, 1.0))
                edges.append(Edge(source=i_g, target=j_g, weight=weight, type="local", cluster_id=cluster.id))

    # Cross-area: each neuron in the smaller area independently tries to bridge once.
    for left in range(len(clusters)):
        for right in range(left + 1, len(clusters)):
            left_ids = clusters[left].neuron_ids
            right_ids = clusters[right].neuron_ids
            if not left_ids or not right_ids:
                continue
            smaller, larger = (left_ids, right_ids) if len(left_ids) <= len(right_ids) else (right_ids, left_ids)
            for i in smaller:
                if rng.random() < p_cross:
                    j = int(rng.choice(larger))
                    edges.append(Edge(source=int(i), target=j, weight=0.4, type="cross", cluster_id=-1))

    # Small-world augmentation: a few extra random cross-area bridges in brain-like mode.
    if brain_like_mode and len(neurons) > 3:
        for _ in range(max(1, int(_SMALL_WORLD_RATE * len(neurons)))):
            i = int(rng.integers(0, len(neurons)))
            j = int(rng.integers(0, len(neurons)))
            if i != j and neurons[i].cluster_id != neurons[j].cluster_id:
                edges.append(Edge(source=i, target=j, weight=0.3, type="cross", cluster_id=-1))

    return edges


def visualize(
    neurons: list[Neuron],
    edges: list[Edge],
    brain_radius: float,
    max_edges_render: int = 4000,
) -> go.Figure:
    fig = go.Figure()
    positions = np.array([n.position for n in neurons], dtype=float)
    neuron_colors = [_CLUSTER_PALETTE[n.cluster_id % len(_CLUSTER_PALETTE)] for n in neurons]

    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode="markers",
            marker={"size": 2.6, "color": neuron_colors, "opacity": 0.9},
            name="neurons",
            hovertemplate="Neuron %{pointNumber}<extra></extra>",
        )
    )

    local_by_cluster: dict[int, tuple[list[float | None], list[float | None], list[float | None]]] = {}
    cross_x: list[float | None] = []
    cross_y: list[float | None] = []
    cross_z: list[float | None] = []

    for edge in edges[:max_edges_render]:
        p1, p2 = positions[edge.source], positions[edge.target]
        xs: list[float | None] = [float(p1[0]), float(p2[0]), None]
        ys: list[float | None] = [float(p1[1]), float(p2[1]), None]
        zs: list[float | None] = [float(p1[2]), float(p2[2]), None]
        if edge.type == "local":
            if edge.cluster_id not in local_by_cluster:
                local_by_cluster[edge.cluster_id] = ([], [], [])
            lx, ly, lz = local_by_cluster[edge.cluster_id]
            lx += xs
            ly += ys
            lz += zs
        else:
            cross_x += xs
            cross_y += ys
            cross_z += zs

    for cid, (lx, ly, lz) in local_by_cluster.items():
        color = _CLUSTER_PALETTE[cid % len(_CLUSTER_PALETTE)]
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz,
            mode="lines",
            line={"width": 1.0, "color": _hex_to_rgba(color, _LOCAL_EDGE_OPACITY)},
            hoverinfo="skip",
            name=f"area {cid}",
        ))

    if cross_x:
        fig.add_trace(go.Scatter3d(
            x=cross_x, y=cross_y, z=cross_z,
            mode="lines",
            line={"width": 1.8, "color": _CROSS_EDGE_COLOR},
            hoverinfo="skip",
            name="cross-area",
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
            ng.label(
                "One unified brain — areas by Voronoi partition, k-nearest local wiring, red cross-area bridges."
            ).classes("text-caption")

            ng.separator()
            ng.label("Structure").classes("text-subtitle2")
            num_areas = slider_field("Number of areas", 2, 12, 6, 1, lambda v: f"{int(v)}")
            neurons_min = slider_field("Min neurons per area", 10, 200, 40, 5, lambda v: f"{int(v)}")
            neurons_max = slider_field("Max neurons per area", 10, 300, 120, 5, lambda v: f"{int(v)}")

            ng.separator()
            ng.label("Connectivity").classes("text-subtitle2")
            k_neighbors = slider_field(
                "Local neighbors (K) — each neuron connects to its K nearest in-area neighbors",
                1, 12, 4, 1, lambda v: f"{int(v)}",
            )
            p_cross = slider_field(
                "Cross-area bridge probability — chance each neuron bridges to another area",
                0.000, 0.150, 0.020, 0.005, lambda v: f"{float(v):.3f}",
            )

            ng.separator()
            ng.label("Global").classes("text-subtitle2")
            brain_like_mode = ng.switch("Brain-like mode  (denser core, small-world bridges)", value=True)
            seed = slider_field("Random seed", 0, 9999, 42, 1, lambda v: f"{int(v)}")

            ng.separator()
            stats = ng.label("Ready").classes("text-caption")
            ng.button("Generate Brain", on_click=lambda: regenerate()).props("color=primary")

        plot_holder = ng.column().style("flex: 1 1 740px; min-height: 80vh;")

    def regenerate() -> None:
        rng = np.random.default_rng(int(seed.value))
        n_areas = int(num_areas.value)
        n_min = min(int(neurons_min.value), int(neurons_max.value))
        n_max = max(int(neurons_min.value), int(neurons_max.value))

        brain_radius = 28.0
        clusters = generate_clusters(n_areas, brain_radius, rng, bool(brain_like_mode.value))

        # Each area contributes a random neuron count in [n_min, n_max].
        area_counts = [int(rng.integers(n_min, n_max + 1)) for _ in range(n_areas)]
        neurons = generate_neurons(sum(area_counts), brain_radius, rng, bool(brain_like_mode.value))
        assign_clusters(neurons, clusters)

        edges = build_connections(
            neurons, clusters,
            k_local=int(k_neighbors.value),
            p_cross=float(p_cross.value),
            rng=rng,
            brain_like_mode=bool(brain_like_mode.value),
        )
        fig = visualize(neurons, edges, brain_radius)

        plot_holder.clear()
        with plot_holder:
            ng.plotly(fig).style("width: 100%; height: 80vh;")

        local_count = sum(1 for e in edges if e.type == "local")
        actual_sizes = sorted(len(c.neuron_ids) for c in clusters)
        stats.text = (
            f"{n_areas} areas · {len(neurons)} neurons  (per area: {actual_sizes}) · "
            f"{local_count} local edges · {len(edges) - local_count} cross-area bridges"
        )

    with container:
        regenerate()


ui()
ng.run(title="Cerebryx Brain Topology Simulator")
