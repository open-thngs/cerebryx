from nicegui import ui
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# Data generation
# -----------------------------

def gaussian_prob(d, sigma):
    return np.exp(-(d**2) / (2 * sigma**2))


def generate_brain(num_clusters, neurons_per_cluster, sigma, p_long, cluster_radius):
    neurons = []
    clusters = []
    edges = []

    neuron_id = 0

    # 1. Create clusters
    for c in range(num_clusters):
        cluster_center = np.random.uniform(-10, 10, size=3)
        cluster = {
            "id": c,
            "center": cluster_center,
            "neurons": []
        }

        # 2. Create neurons in cluster
        for i in range(neurons_per_cluster):
            pos = cluster_center + np.random.normal(scale=cluster_radius, size=3)
            neurons.append({
                "id": neuron_id,
                "pos": pos,
                "cluster": c
            })
            cluster["neurons"].append(neuron_id)
            neuron_id += 1

        clusters.append(cluster)

    # 3. Local connections (within cluster)
    for cluster in clusters:
        ids = cluster["neurons"]
        for i in ids:
            for j in ids:
                if i >= j:
                    continue

                d = np.linalg.norm(neurons[i]["pos"] - neurons[j]["pos"])
                p = gaussian_prob(d, sigma)

                if np.random.rand() < p:
                    edges.append((i, j, p))

    # 4. Long-range connections (between clusters)
    all_ids = list(range(len(neurons)))

    for i in all_ids:
        if np.random.rand() < p_long:
            j = np.random.choice(all_ids)
            if i != j:
                d = np.linalg.norm(neurons[i]["pos"] - neurons[j]["pos"])
                w = gaussian_prob(d, sigma)
                edges.append((i, j, w))

    return neurons, clusters, edges


# -----------------------------
# Visualization
# -----------------------------

def plot_brain(neurons, clusters, edges):
    fig = go.Figure()

    # neurons
    xs = [n["pos"][0] for n in neurons]
    ys = [n["pos"][1] for n in neurons]
    zs = [n["pos"][2] for n in neurons]
    colors = [n["cluster"] for n in neurons]

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=3, color=colors, colorscale='Viridis'),
        name='neurons'
    ))

    # edges: collect into two traces to reduce redraw overhead and flicker
    local_x, local_y, local_z = [], [], []
    long_x, long_y, long_z = [], [], []

    for i, j, _w in edges[:2000]:  # limit for performance
        p1 = neurons[i]["pos"]
        p2 = neurons[j]["pos"]
        if neurons[i]["cluster"] == neurons[j]["cluster"]:
            local_x.extend([p1[0], p2[0], None])
            local_y.extend([p1[1], p2[1], None])
            local_z.extend([p1[2], p2[2], None])
        else:
            long_x.extend([p1[0], p2[0], None])
            long_y.extend([p1[1], p2[1], None])
            long_z.extend([p1[2], p2[2], None])

    if local_x:
        fig.add_trace(go.Scatter3d(
            x=local_x,
            y=local_y,
            z=local_z,
            mode='lines',
            line=dict(width=1.2, color='rgba(24, 110, 52, 0.6)'),
            name='local',
            showlegend=False
        ))

    if long_x:
        fig.add_trace(go.Scatter3d(
            x=long_x,
            y=long_y,
            z=long_z,
            mode='lines',
            line=dict(width=1.5, color='rgba(255, 90, 90, 0.55)'),
            name='long-range',
            showlegend=False
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        uirevision='brain-topology',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False),
            zaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False),
            bgcolor='rgba(0,0,0,0)'
        )
    )

    return fig


# -----------------------------
# UI
# -----------------------------


def add_slider(label_text, tooltip_text, min_value, max_value, value, step, formatter=None):
    with ui.column().style('width: 520px; gap: 0.25rem;'):
        with ui.row().classes('w-full items-center justify-between'):
            ui.label(label_text).tooltip(tooltip_text)
            value_label = ui.label()
        slider = ui.slider(min=min_value, max=max_value, value=value, step=step).classes('w-full')
        if formatter is None:
            value_label.bind_text_from(slider, 'value')
        else:
            value_label.bind_text_from(slider, 'value', backward=formatter)
    return slider


with ui.row().style('width: 100%; gap: 24px; align-items: flex-start; flex-wrap: wrap;'):
    with ui.column().style('flex: 0 0 560px; max-width: 560px; gap: 0.75rem;'):
        ui.label("Neuromorphic Brain Topology Simulator").classes("text-h5")
        num_clusters = add_slider(
            'Clusters',
            'Anzahl der Cluster im simulierten Gehirn',
            1,
            10,
            3,
            1,
        )
        neurons_per_cluster = add_slider(
            'Neurons per Cluster',
            'Wie viele Neuronen pro Cluster erzeugt werden',
            5,
            200,
            30,
            1,
        )
        sigma = add_slider(
            'Local Connectivity Sigma',
            'Streuung der lokalen Verbindungen innerhalb eines Clusters',
            0.5,
            10.0,
            3.0,
            0.1,
            formatter=lambda v: f'{float(v):.1f}',
        )
        p_long = add_slider(
            'Long-range connection probability',
            'Wahrscheinlichkeit für Verbindungen zwischen Clustern (in Prozent)',
            0,
            100,
            5,
            1,
            formatter=lambda v: f'{int(v)}%',
        )
        cluster_radius = add_slider(
            'Cluster spread',
            'Wie weit die Neuronen eines Clusters verteilt sind',
            0.5,
            5.0,
            2.0,
            0.1,
            formatter=lambda v: f'{float(v):.1f}',
        )
        ui.button("Generate Brain", on_click=lambda: regenerate())

    plot_container = ui.column().style('flex: 1 1 700px; min-height: 78vh;')


def regenerate():
    # Mappe p_long von Prozent auf 0.0-1.0
    neurons, clusters, edges = generate_brain(
        num_clusters.value,
        neurons_per_cluster.value,
        sigma.value,
        p_long.value / 100.0,
        cluster_radius.value
    )

    fig = plot_brain(neurons, clusters, edges)

    plot_container.clear()
    with plot_container:
        ui.plotly(fig).style('width: 100%; height: 78vh;')


regenerate()

ui.run()
