from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import seaborn as sns

sys.path.append(Path(__file__).resolve().parents[3].as_posix())

from utils import export_to_obj

from src.surface.cylinder import CylinderSurface

sns.set_style(style="whitegrid")
COLORS = {
    "normal": "#e74c3c",
}


def visualize_cylinder(surface: CylinderSurface, num_u_segments: int = 32, num_v_segments: int = 2, figsize: tuple = (10, 8)) -> None:
    vertices, faces, normals = surface.get_regular_mesh(num_u_segments, num_v_segments)
    vertices_np = vertices.numpy()
    faces_np = faces.numpy()
    normals_np = normals.numpy()
    uvs = surface.get_regular_uv(num_u_segments, num_v_segments)
    uvs = (uvs - uvs.amin(0)) / (uvs.amax(0) - uvs.amin(0))
    uvs_np = uvs.numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    colors = np.concatenate([uvs_np, np.full_like(uvs_np[..., 0 : 1], 0.5)], axis=-1)
    mesh = Poly3DCollection(vertices_np[faces_np], edgecolors="none", facecolors=colors[faces_np].mean(-2), alpha=0.8, linewidths=0.5)
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
    ax.set_title(f"Cylinder Surface (Radius = {surface.radius}, Height = {surface.height})", fontsize=14)

    plt.show()


if __name__ == "__main__":
    num_u_segments, num_v_segments = 32, 5
    cylinder = CylinderSurface(radius=0.25, height=1.0)
    visualize_cylinder(cylinder, num_u_segments, num_v_segments)
    # export_to_obj("cylinder.obj", cylinder, num_u_segments, num_v_segments)
