from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import plotly.graph_objects as go
from nicegui import ui as ng

# --- Cluster placement ---
_CLUSTER_PLACEMENT_ATTEMPTS = 300
_CLUSTER_PLACEMENT_FRAC = 0.88
_BETA_A = 2.2
_BETA_B = 4.8


# --- Visualization ---
_LOCAL_EDGE_OPACITY = 0.35
_BUBBLE_OPACITY = 0.07
_SPHERE_THETA = 20
_SPHERE_PHI = 14
# Red is reserved for cross-area bridges — no palette entry may be a red shade.
_CROSS_EDGE_COLOR = "rgba(215, 55, 55, 0.75)"
_CLUSTER_PALETTE = [
    "#2EC4B6",  # teal
    "#FF9F1C",  # amber
    "#5BC0EB",  # sky cyan
    "#8338EC",  # violet
    "#06D6A0",  # emerald
    "#F15BB5",  # pink
    "#8AC926",  # lime
    "#3A86FF",  # cornflower blue
    "#FFCA3A",  # yellow
    "#B5179E",  # magenta
]

_SIDEBAR_BG = "#0e0e1a"
_PAGE_BG    = "#080810"
_ACCENT     = "#2EC4B6"


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


def _shell_radius(rng: np.random.Generator, min_r: float, max_r: float, brain_like: bool) -> float:
    """Sample a radius inside a spherical shell [min_r, max_r]."""
    if max_r <= min_r:
        return max_r
    if brain_like:
        t = float(rng.beta(_BETA_A, _BETA_B))
        return min_r + t * (max_r - min_r)
    u = float(rng.random())
    return (min_r ** 3 + u * (max_r ** 3 - min_r ** 3)) ** (1.0 / 3.0)


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
    min_r = 0.0
    max_r = brain_radius * _CLUSTER_PLACEMENT_FRAC
    min_separation = max(1.5, brain_radius / (2.0 * num_clusters))

    clusters: list[Cluster] = []
    for cluster_id in range(num_clusters):
        placed = False
        for _ in range(_CLUSTER_PLACEMENT_ATTEMPTS):
            r = _shell_radius(rng, min_r, max_r, brain_like_mode)
            center = _random_unit(rng) * max(0.5, r)
            if not clusters or min(np.linalg.norm(center - c.center) for c in clusters) >= min_separation:
                placed = True
                break
        if not placed:
            center = _random_unit(rng) * max(0.5, (min_r + max_r) / 2)
        clusters.append(Cluster(id=cluster_id, center=center))

    return clusters


def generate_neurons(
    num_neurons: int,
    brain_radius: float,
    rng: np.random.Generator,
    brain_like_mode: bool,
) -> list[Neuron]:
    min_r = 0.0
    max_r = brain_radius

    neurons: list[Neuron] = []
    for nid in range(num_neurons):
        r = _shell_radius(rng, min_r, max_r, brain_like_mode)
        neurons.append(Neuron(id=nid, cluster_id=0, position=_random_unit(rng) * r))
    return neurons


def assign_clusters(neurons: list[Neuron], clusters: list[Cluster]) -> None:
    """Voronoi partition: assign each neuron to its nearest cluster center."""
    for cluster in clusters:
        cluster.neuron_ids = []
    centers   = np.array([c.center for c in clusters], dtype=float)
    positions = np.array([n.position for n in neurons], dtype=float)
    dists     = np.linalg.norm(positions[:, None, :] - centers[None, :, :], axis=2)
    for neuron_idx, cluster_idx in enumerate(np.argmin(dists, axis=1)):
        cid = int(cluster_idx)
        neurons[neuron_idx].cluster_id = clusters[cid].id
        clusters[cid].neuron_ids.append(neuron_idx)


def build_connections(neurons: list[Neuron], k: int) -> list[Edge]:
    """Global k-nearest-neighbor pass over all neurons, area-blind.

    Every neuron connects to its k spatially closest neighbors regardless of
    which Voronoi area they belong to. Edges that happen to cross an area
    boundary are classified as 'cross' and colored red at render time; all
    others are 'local'. No explicit cross-area bridges or small-world wiring.
    """
    n = len(neurons)
    if n < 2:
        return []

    positions = np.array([n.position for n in neurons], dtype=float)
    dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)

    k = min(k, n - 1)
    edges: list[Edge] = []
    seen: set[tuple[int, int]] = set()

    for i in range(n):
        for j in map(int, np.argsort(dists[i])[:k]):
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            d = float(dists[i, j])
            weight = float(np.clip(1.0 / (1.0 + d * 0.3), 0.05, 1.0))
            same = neurons[i].cluster_id == neurons[j].cluster_id
            edges.append(Edge(
                source=i, target=j, weight=weight,
                type="local" if same else "cross",
                cluster_id=neurons[i].cluster_id if same else -1,
            ))

    return edges


def _make_sphere_surface(center: np.ndarray, radius: float, color: str) -> go.Surface:
    theta = np.linspace(0, 2 * np.pi, _SPHERE_THETA)
    phi   = np.linspace(0, np.pi, _SPHERE_PHI)
    x = center[0] + radius * np.outer(np.cos(theta), np.sin(phi))
    y = center[1] + radius * np.outer(np.sin(theta), np.sin(phi))
    z = center[2] + radius * np.outer(np.ones_like(theta), np.cos(phi))
    return go.Surface(
        x=x, y=y, z=z,
        opacity=_BUBBLE_OPACITY,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        hoverinfo="skip",
        showlegend=False,
        name="bubble",
    )


def _make_area_hull(
    cluster: Cluster, positions: np.ndarray, color: str
) -> go.Mesh3d | go.Surface | None:
    """Organic convex-hull bubble following the actual shape of the neuron cloud."""
    if not cluster.neuron_ids:
        return None
    pts = positions[cluster.neuron_ids]
    if len(pts) < 4:
        center = pts.mean(axis=0)
        return _make_sphere_surface(center, 5.0, color)
    centroid = pts.mean(axis=0)
    expanded = centroid + (pts - centroid) * 1.18
    return go.Mesh3d(
        x=expanded[:, 0], y=expanded[:, 1], z=expanded[:, 2],
        alphahull=0,
        opacity=_BUBBLE_OPACITY,
        color=color,
        hoverinfo="skip",
        showlegend=False,
        name="hull",
        flatshading=True,
    )


def visualize(
    neurons: list[Neuron],
    edges: list[Edge],
    clusters: list[Cluster],
    brain_radius: float,
    show_bubbles: bool = False,
    show_bridges: bool = True,
    hidden_areas: set[int] | None = None,
    max_edges_render: int = 4000,
) -> go.Figure:
    if hidden_areas is None:
        hidden_areas = set()

    fig = go.Figure()
    positions = np.array([n.position for n in neurons], dtype=float)

    vis_neurons = [n for n in neurons if n.cluster_id not in hidden_areas]
    vis_ids     = {n.id for n in vis_neurons}

    # Bubbles rendered first so they sit behind neurons and edges.
    if show_bubbles:
        for cluster in clusters:
            if cluster.id in hidden_areas:
                continue
            color = _CLUSTER_PALETTE[cluster.id % len(_CLUSTER_PALETTE)]
            hull = _make_area_hull(cluster, positions, color)
            if hull is not None:
                fig.add_trace(hull)

    if vis_neurons:
        vis_pos    = np.array([n.position for n in vis_neurons], dtype=float)
        vis_colors = [_CLUSTER_PALETTE[n.cluster_id % len(_CLUSTER_PALETTE)] for n in vis_neurons]
        fig.add_trace(go.Scatter3d(
            x=vis_pos[:, 0], y=vis_pos[:, 1], z=vis_pos[:, 2],
            mode="markers",
            marker={"size": 2.8, "color": vis_colors, "opacity": 0.92},
            name="neurons",
            hovertemplate="Neuron %{pointNumber}<extra></extra>",
        ))

    # Collect local edges per cluster and all cross-area edges, then apply a
    # per-cluster render limit so high-k settings never starve later clusters.
    local_segs_by_cid: dict[int, list[tuple]] = {}
    cross_segs: list[tuple] = []
    for edge in edges:
        if edge.source not in vis_ids or edge.target not in vis_ids:
            continue
        p1, p2 = positions[edge.source], positions[edge.target]
        seg = (
            [float(p1[0]), float(p2[0]), None],
            [float(p1[1]), float(p2[1]), None],
            [float(p1[2]), float(p2[2]), None],
        )
        if edge.type == "local":
            local_segs_by_cid.setdefault(edge.cluster_id, []).append(seg)
        else:
            cross_segs.append(seg)

    n_areas = max(1, len(local_segs_by_cid))
    per_cluster_limit = max(300, max_edges_render // n_areas)

    local_by_cluster: dict[int, tuple[list[float | None], list[float | None], list[float | None]]] = {}
    for cid, segs in local_segs_by_cid.items():
        limited = segs[:per_cluster_limit]
        local_by_cluster[cid] = (
            [v for s in limited for v in s[0]],
            [v for s in limited for v in s[1]],
            [v for s in limited for v in s[2]],
        )

    cross_x: list[float | None] = [v for s in cross_segs for v in s[0]]
    cross_y: list[float | None] = [v for s in cross_segs for v in s[1]]
    cross_z: list[float | None] = [v for s in cross_segs for v in s[2]]

    for cid, (lx, ly, lz) in local_by_cluster.items():
        color = _CLUSTER_PALETTE[cid % len(_CLUSTER_PALETTE)]
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz,
            mode="lines",
            line={"width": 1.0, "color": _hex_to_rgba(color, _LOCAL_EDGE_OPACITY)},
            hoverinfo="skip",
            name=f"area {cid}",
        ))

    if cross_x and show_bridges:
        fig.add_trace(go.Scatter3d(
            x=cross_x, y=cross_y, z=cross_z,
            mode="lines",
            line={"width": 2.0, "color": _CROSS_EDGE_COLOR},
            hoverinfo="skip",
            name="cross-area",
        ))

    r = brain_radius
    axis_cfg = {"visible": False, "range": [-r, r], "showgrid": False, "zeroline": False}
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        uirevision=1,
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
    ng.dark_mode(value=True)
    ng.add_head_html(f"""<style>
      html, body, .q-page, .q-layout {{
        background: {_PAGE_BG} !important;
        margin: 0; padding: 0;
      }}
      .nicegui-content {{
        padding: 0 !important;
        max-width: 100% !important;
      }}
      .sidebar {{
        background: {_SIDEBAR_BG};
        border-right: 1px solid #1c1c2e;
      }}
      .section-label {{
        font-size: 0.58rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        color: {_ACCENT};
        padding-top: 0.55rem;
        padding-bottom: 0.05rem;
      }}
      .stat-text {{
        font-size: 0.63rem;
        color: #40405a;
        line-height: 1.75;
        font-variant-numeric: tabular-nums;
      }}
      ::-webkit-scrollbar {{ width: 4px; }}
      ::-webkit-scrollbar-track {{ background: transparent; }}
      ::-webkit-scrollbar-thumb {{ background: #222235; border-radius: 2px; }}
    </style>""")

    def divider() -> None:
        ng.separator().style("background:#1a1a2c; height:1px; border:none; margin:0.25rem 0;")

    def section(text: str) -> None:
        ng.label(text).classes("section-label")

    def slider_field(
        label: str,
        min_value: float,
        max_value: float,
        value: float,
        step: float,
        formatter: Callable[[float], str],
    ):
        with ng.column().style("gap:0.05rem;"):
            with ng.row().classes("w-full items-center justify-between").style("gap:0.5rem;"):
                ng.label(label).style("font-size:0.71rem; color:#606080;")
                value_label = ng.label().style(
                    "font-size:0.71rem; color:#c8c8e0; font-weight:600; "
                    "font-variant-numeric:tabular-nums; white-space:nowrap;"
                )
            slider = (
                ng.slider(min=min_value, max=max_value, value=value, step=step)
                .classes("w-full")
                .props("color=teal dense")
            )
            value_label.bind_text_from(slider, "value", backward=formatter)
        return slider

    def toggle_row(label: str, sublabel: str, value: bool, on_change=None):
        with ng.row().classes("items-center justify-between w-full").style("padding:0.1rem 0;"):
            with ng.column().style("gap:0.04rem;"):
                ng.label(label).style("font-size:0.71rem; color:#606080;")
                ng.label(sublabel).style("font-size:0.6rem; color:#2e2e48;")
            sw = ng.switch("", value=value, on_change=on_change).props("color=teal dense")
        return sw

    # ── Full-viewport layout ──────────────────────────────────────────────────
    with ng.row().style(f"height:100vh; width:100%; overflow:hidden; gap:0; background:{_PAGE_BG};"):

        # ── Sidebar ───────────────────────────────────────────────────────────
        with ng.column().classes("sidebar").style(
            "width:300px; min-width:300px; height:100vh; overflow-y:auto; "
            "padding:1.3rem 1.05rem 1.1rem; gap:0.3rem;"
        ):
            ng.label("CEREBRYX").style(
                f"font-size:1.25rem; font-weight:800; letter-spacing:0.24em; color:{_ACCENT};"
            )
            ng.label("Brain Topology Simulator").style(
                "font-size:0.65rem; color:#282840; margin-top:-0.15rem; margin-bottom:0.4rem;"
            )

            divider()
            section("STRUCTURE")
            num_areas   = slider_field("Areas", 2, 12, 6, 1, lambda v: f"{int(v)}")
            neurons_min = slider_field("Min neurons / area", 10, 200, 40, 5, lambda v: f"{int(v)}")
            neurons_max = slider_field("Max neurons / area", 10, 300, 120, 5, lambda v: f"{int(v)}")

            divider()
            section("CONNECTIVITY")
            k_neighbors = slider_field("Neighbors  K", 1, 12, 4, 1, lambda v: f"{int(v)}")

            divider()
            section("OPTIONS")
            brain_like_mode = toggle_row(
                "Brain-like mode", "Dense core · small-world bridges", True,
            )
            seed = slider_field("Random seed", 0, 9999, 42, 1, lambda v: f"{int(v)}")

            divider()
            ng.button("GENERATE BRAIN", on_click=lambda: regenerate()).classes("w-full").props(
                "color=teal unelevated"
            ).style("font-weight:700; letter-spacing:0.07em; margin-top:0.2rem;")
            stats = ng.label("—").classes("stat-text").style("margin-top:0.3rem;")

        # ── 3-D plot + floating legend (siblings inside a position:relative wrapper) ─
        with ng.column().style(
            f"flex:1 1 0; height:100vh; overflow:hidden; "
            f"background:{_PAGE_BG}; position:relative;"
        ):
            plot_container   = ng.column().style("width:100%; height:100%;")
            legend_container = ng.column().style(
                "position:absolute; top:1.1rem; right:1.1rem; z-index:1000; "
                "background:rgba(10,10,22,0.86); border:1px solid #252538; "
                "border-radius:10px; padding:0.65rem 0.75rem 0.7rem; gap:0.22rem; "
                "min-width:164px;"
            )

    state: dict = {}

    def toggle_area(cid: int) -> None:
        hidden: set[int] = state.get("hidden_areas", set())
        if cid in hidden:
            hidden.discard(cid)
        else:
            hidden.add(cid)
        state["hidden_areas"] = hidden
        rerender()

    def toggle_bridges() -> None:
        state["show_bridges"] = not state.get("show_bridges", True)
        rerender()

    def toggle_bubbles() -> None:
        state["show_bubbles"] = not state.get("show_bubbles", False)
        rerender()

    def _rebuild_legend() -> None:
        hidden = state.get("hidden_areas", set())
        legend_container.clear()
        with legend_container:
            # Header row: title + bridge + bubble icon toggles
            with ng.row().classes("items-center").style("gap:0.25rem; padding-bottom:0.22rem;"):
                ng.label("AREAS").style(
                    f"font-size:0.55rem; font-weight:700; letter-spacing:0.16em; "
                    f"color:{_ACCENT}; flex:1;"
                )
                bridges_on = state.get("show_bridges", True)
                ng.button(icon="share", on_click=lambda: toggle_bridges()).props(
                    "flat dense size=xs round"
                ).style(f"color:{_ACCENT if bridges_on else '#2a2a40'};").tooltip(
                    "Inter-area connections"
                )
                bubbles_on = state.get("show_bubbles", False)
                ng.button(icon="bubble_chart", on_click=lambda: toggle_bubbles()).props(
                    "flat dense size=xs round"
                ).style(f"color:{_ACCENT if bubbles_on else '#2a2a40'};").tooltip(
                    "Area bubbles"
                )
            ng.separator().style(
                "background:#1a1a2c; height:1px; border:none; margin:0.08rem 0 0.18rem;"
            )
            for cluster in state["clusters"]:
                color     = _CLUSTER_PALETTE[cluster.id % len(_CLUSTER_PALETTE)]
                cid       = cluster.id
                is_hidden = cid in hidden
                n_count   = len(cluster.neuron_ids)
                dot_op    = "0.18" if is_hidden else "1"
                lbl_col   = "#2c2c44" if is_hidden else "#aeaec8"
                cnt_col   = "#242438" if is_hidden else "#3e3e58"
                btn_col   = "#252538" if is_hidden else _ACCENT
                with ng.row().classes("items-center").style("gap:0.38rem;"):
                    ng.button(
                        icon="visibility_off" if is_hidden else "visibility",
                        on_click=lambda c=cid: toggle_area(c),
                    ).props("flat dense size=xs round").style(f"color:{btn_col};")
                    ng.html(
                        f'<div style="width:9px;height:9px;border-radius:50%;'
                        f'background:{color};opacity:{dot_op};flex-shrink:0;"></div>'
                    )
                    ng.label(f"Area {cid + 1}").style(
                        f"font-size:0.7rem; color:{lbl_col}; flex:1; min-width:3.2rem;"
                    )
                    ng.label(str(n_count)).style(f"font-size:0.62rem; color:{cnt_col};")

    def rerender(*, hard: bool = False) -> None:
        if not state:
            return
        fig = visualize(
            state["neurons"],
            state["edges"],
            state["clusters"],
            state["brain_radius"],
            show_bubbles=state.get("show_bubbles", False),
            show_bridges=state.get("show_bridges", True),
            hidden_areas=state.get("hidden_areas"),
        )
        # Inject the saved camera so both Plotly.react and Plotly.newPlot
        # render at the correct position (guards against the async-load race
        # where last_options is still null on the first soft render).
        if state.get("camera"):
            fig.update_layout(scene_camera=state["camera"])

        if hard or "plot_el" not in state:
            plot_container.clear()
            with plot_container:
                state["plot_el"] = ng.plotly(fig).style("width:100%; height:100vh;")

            def _on_relayout(e: object) -> None:
                if isinstance(e.args, dict) and "scene.camera" in e.args:
                    state["camera"] = e.args["scene.camera"]

            state["plot_el"].on("plotly_relayout", _on_relayout)
        else:
            state["plot_el"].update_figure(fig)
        _rebuild_legend()

    def regenerate() -> None:
        rng     = np.random.default_rng(int(seed.value))
        n_areas = int(num_areas.value)
        n_min   = min(int(neurons_min.value), int(neurons_max.value))
        n_max   = max(int(neurons_min.value), int(neurons_max.value))
        brain_r = 28.0

        clusters = generate_clusters(n_areas, brain_r, rng, bool(brain_like_mode.value))
        total    = sum(int(rng.integers(n_min, n_max + 1)) for _ in range(n_areas))
        neurons  = generate_neurons(total, brain_r, rng, bool(brain_like_mode.value))
        assign_clusters(neurons, clusters)
        edges    = build_connections(neurons, k=int(k_neighbors.value))

        # Preserve view toggles across generations; reset area visibility.
        state.setdefault("show_bridges", True)
        state.setdefault("show_bubbles", False)
        state.update({
            "neurons":      neurons,
            "clusters":     clusters,
            "edges":        edges,
            "brain_radius": brain_r,
            "hidden_areas": set(),
            "camera":       None,
        })
        rerender(hard=True)

        local_count = sum(1 for e in edges if e.type == "local")
        sizes = sorted(len(c.neuron_ids) for c in clusters)
        stats.text = (
            f"{n_areas} areas  ·  {total} neurons\n"
            f"{local_count} local  ·  {len(edges) - local_count} cross-area\n"
            f"per area: {sizes}"
        )

    regenerate()


ui()
ng.run(title="Cerebryx")
