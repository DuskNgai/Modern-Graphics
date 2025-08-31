from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns

sys.path.append(Path(__file__).resolve().parents[3].as_posix())

from src.surface.sphere import SphereSurface

sns.set_style(style="whitegrid")
COLORS = {
    "surface": "#3498db",
    "normal": "#e74c3c",
    "wireframe": "#2c3e50",
}


def visualize_sphere(surface: SphereSurface, num_segments_per_edge: int = 16, figsize: tuple = (10, 8)) -> None:
    vertices, faces, normals = surface.get_regular_mesh(num_segments_per_edge)
    vertices_np = vertices.numpy()
    faces_np = faces.numpy()
    normals_np = normals.numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    mesh = Poly3DCollection(vertices_np[faces_np], alpha=0.8, linewidths=0.5, edgecolors=COLORS["wireframe"], facecolors=COLORS["surface"])
    ax.add_collection3d(mesh)

    quiver_plot = ax.quiver(*vertices_np.T, *normals_np.T, color=COLORS["normal"], length=0.1, arrow_length_ratio=0.2, label="Normals")
    quiver_plot.set_visible(False)

    ax_button = plt.axes([0.3, 0.05, 0.4, 0.06])
    toggle_button = Button(ax_button, "Toggle Normals")

    def toggle_normals(event):
        visible = quiver_plot.get_visible()
        quiver_plot.set_visible(not visible)
        plt.draw()

    toggle_button.on_clicked(toggle_normals)

    ax.legend()
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=45)
    ax.set_title(f"Sphere Surface (Radius = {surface.radius})", fontsize=14)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sphere = SphereSurface(radius=1.0)
    visualize_sphere(sphere, num_segments_per_edge=16)
