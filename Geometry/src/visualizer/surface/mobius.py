from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import seaborn as sns

sys.path.append(Path(__file__).resolve().parents[3].as_posix())

from src.surface.mobius import MobiusSurface
from utils import export_to_obj

sns.set_style(style="whitegrid")
COLORS = {
    "normal": "#e74c3c",
}


def visualize_mobius(surface: MobiusSurface, num_u_segments: int = 24, num_v_segments: int = 24, figsize: tuple = (10, 8)) -> None:
    vertices, faces, normals = surface.get_regular_mesh(num_u_segments, num_v_segments)
    vertices_np = vertices.numpy()
    faces_np = faces.numpy()
    normals_np = normals.numpy()
    u, v = surface.get_regular_uv(num_u_segments, num_v_segments)
    u = (u - u.min()) / (u.max() - u.min())
    v = (v - v.min()) / (v.max() - v.min())
    uvs_np = np.stack([u.numpy(), v.numpy()], axis=-1).reshape(-1, 2)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    colors = np.concatenate([uvs_np, np.full_like(uvs_np[..., 0 : 1], 0.5)], axis=-1)
    mesh = Poly3DCollection(vertices_np[faces_np], edgecolors="none", facecolors=colors[faces_np].mean(-2), alpha=0.8, linewidths=0.5)
    ax.add_collection3d(mesh)

    quiver_plot = ax.quiver(*vertices_np.T, *normals_np.T, color=COLORS["normal"], length=0.15, arrow_length_ratio=0.3, label="Normals")
    quiver_plot.set_visible(False)

    ax_button = plt.axes([0.35, 0.02, 0.3, 0.05])
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
    ax.set_title(f"Mobius Surface (Radius = {surface.radius}, Width = {surface.width})", fontsize=14)

    plt.show()


if __name__ == "__main__":
    num_u_segments, num_v_segments = 24, 12
    mobius = MobiusSurface(radius=1.4, width=1.0)
    visualize_mobius(mobius, num_u_segments, num_v_segments)
    # export_to_obj("mobius.obj", mobius, num_u_segments, num_v_segments)
