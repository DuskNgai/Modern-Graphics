from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import seaborn as sns

sys.path.append(Path(__file__).resolve().parents[3].as_posix())

from src.surface import BezierTriangleSurface

sns.set_style(style="whitegrid")
COLORS = {
    "control": "#3498db",
    "normal": "#e74c3c",
}


def draw_control_net(ax: plt.Axes, control_points: np.ndarray, degree: int, **kwargs) -> list:
    artists = []

    scatter_plot = ax.scatter(*control_points.T, **kwargs)
    artists.append(scatter_plot)

    edges = []
    for i in range(degree):
        l, r = i * (i + 1) // 2, (i + 1) * (i + 2) // 2
        for j in range(l, r):
            edges.append([j, j + i + 1])
            edges.append([j + i + 1, j + i + 2])
            edges.append([j + i + 2, j])

    plot_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ['s', 'label']
    }
    for edge in edges:
        p1 = control_points[edge[0]]
        p2 = control_points[edge[1]]
        x, y, z = zip(p1, p2)
        line_plot = ax.plot(x, y, z, **plot_kwargs)
        artists.extend(line_plot)

    return artists


def visualize_bezier_triangle(surface: BezierTriangleSurface, num_segments_per_edge: int = 11, figsize: tuple = (10, 8)) -> None:
    control_points = surface.control_point.numpy()
    vertices, faces, normals = surface.get_regular_mesh(num_segments_per_edge)
    vertices_np = vertices.numpy()
    faces_np = faces.numpy()
    normals_np = normals.numpy()
    uvws_np = surface.get_regular_uvw(num_segments_per_edge).numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    colors = np.concatenate([uvws_np, np.full_like(uvws_np[..., 0 : 1], 0.75)], axis=-1)
    mesh = Poly3DCollection(vertices_np[faces_np], edgecolors="none", facecolors=colors[faces_np].mean(-2))
    ax.add_collection3d(mesh)

    draw_control_net(ax, control_points, surface.degree, color=COLORS["control"], s=50, linewidth=1, alpha=0.75, label="Control Points")

    quiver_plot = ax.quiver(*vertices_np.T, *normals_np.T, color=COLORS["normal"], length=0.1, arrow_length_ratio=0.1, label="Normals")
    quiver_plot.set_visible(False)

    ax_button = plt.axes([0.3, 0.05, 0.4, 0.06])
    toggle_button = Button(ax_button, "Toggle Normals")

    def toggle_normals(event):
        visible = quiver_plot.get_visible()
        quiver_plot.set_visible(not visible)
        plt.draw()

    toggle_button.on_clicked(toggle_normals)

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=45, azim=-135)
    ax.set_title(f"Bezier Triangle (Degree {surface.degree})", fontsize=14)

    plt.show()


def visualize_split_bezier_triangle(surface: BezierTriangleSurface, num_segments_per_edge: int = 11, figsize: tuple = (12, 10)) -> None:
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(bottom=0.25)
    ax = fig.add_subplot(111, projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    control_points = surface.control_point.numpy()

    draw_control_net(ax, control_points, surface.degree, color="gray", s=20, linewidth=1, alpha=0.5, label="Original Control Points")

    plotted_artists = []

    def update(val):
        nonlocal plotted_artists
        for collection in plotted_artists:
            collection.remove()
        plotted_artists = []

        r, s, t = slider_r.val, slider_s.val, slider_t.val

        colors = ['#E27D60', '#85CDCA', '#E8A87C', '#C38D9D']
        alphas = [0.8, 0.8, 0.8, 0.8]

        for i, sub_surface in enumerate(surface.split(r, s, t)):
            vertices = sub_surface.get_regular_vertex(num_segments_per_edge).numpy()
            faces = sub_surface.get_regular_face(num_segments_per_edge).numpy()
            collection = ax.plot_trisurf(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                triangles=faces,
                color=colors[i],
                alpha=alphas[i],
                edgecolor="none",
            )
            plotted_artists.append(collection)

            sub_control_points = sub_surface.control_point.numpy()
            sub_net_artists = draw_control_net(ax, sub_control_points, sub_surface.degree, color=colors[i], s=25, linewidth=1.5, alpha=0.9)
            plotted_artists.extend(sub_net_artists)

        fig.canvas.draw_idle()

    ax_r = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    ax_s = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    ax_t = fig.add_axes([0.25, 0.05, 0.65, 0.03])

    slider_r = Slider(ax=ax_r, label='r (v-w edge)', valmin=0.0, valmax=1.0, valinit=0.5)
    slider_s = Slider(ax=ax_s, label='s (w-u edge)', valmin=0.0, valmax=1.0, valinit=0.5)
    slider_t = Slider(ax=ax_t, label='t (u-v edge)', valmin=0.0, valmax=1.0, valinit=0.5)

    slider_r.on_changed(update)
    slider_s.on_changed(update)
    slider_t.on_changed(update)

    update(None)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=45, azim=-135)
    ax.legend()
    ax.set_title('Interactive Bezier Triangle Splitting')

    plt.show()


if __name__ == "__main__":
    degree = 2
    control_points = [[1, 1, 1], [0, 1, 0.5], [1, 0, 0.5], [0, 0.5, 0], [0, 0, 0], [0.5, 0, 0]]
    degree = 3
    control_points = [
        [1, 1, 1],
        [0, 1, 2 / 3],
        [1, 0, 2 / 3],
        [0, 0.5, 1 / 3],
        [0.0, 0.0, 1 / 3],
        [0.75, 0, 1 / 3],
        [0, 1, 0],
        [-0.5, 0.5, -1 / 3],
        [0.5, -0.5, -1 / 3],
        [1, 0, 0],
    ]
    surface = BezierTriangleSurface(degree, control_points)
    visualize_bezier_triangle(surface, num_segments_per_edge=11)
    visualize_split_bezier_triangle(surface, num_segments_per_edge=11)
