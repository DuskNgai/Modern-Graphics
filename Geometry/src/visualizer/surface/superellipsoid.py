from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import seaborn as sns

sys.path.append(Path(__file__).resolve().parents[3].as_posix())

from src.surface.superellipsoid import SuperellipsoidSurface

sns.set_style(style="whitegrid")
COLORS = {
    "normal": "#e74c3c",
}


def visualize_superellipsoid(surface: SuperellipsoidSurface, num_u_segments: int = 48, num_v_segments: int = 24, figsize: tuple = (10, 8)) -> None:
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

    colors = np.concatenate([uvs_np, np.full_like(uvs_np[..., 0:1], 0.5)], axis=-1)
    mesh = Poly3DCollection(vertices_np[faces_np], edgecolors="none", facecolors=colors[faces_np].mean(-2), alpha=0.8, linewidths=0.5)
    ax.add_collection3d(mesh)

    quiver_plot = ax.quiver(*vertices_np.T, *normals_np.T, color=COLORS["normal"], length=0.1, arrow_length_ratio=0.2, label="Normals")
    quiver_plot.set_visible(False)

    # Sliders for e1 and e2
    ax_e1 = plt.axes([0.15, 0.02, 0.3, 0.03])
    ax_e2 = plt.axes([0.55, 0.02, 0.3, 0.03])
    slider_e1 = Slider(ax_e1, "e1", 0.1, 5.0, valinit=surface.e1)
    slider_e2 = Slider(ax_e2, "e2", 0.1, 5.0, valinit=surface.e2)

    ax_button = plt.axes([0.4, 0.06, 0.2, 0.04])
    toggle_button = Button(ax_button, "Toggle Normals")

    quiver_visible = [False]

    def toggle_normals(event):
        quiver_visible[0] = not quiver_visible[0]
        quiver_plot.set_visible(quiver_visible[0])
        plt.draw()

    toggle_button.on_clicked(toggle_normals)

    def update(val):
        # update surface parameters
        surface.e1 = float(slider_e1.val)
        surface.e2 = float(slider_e2.val)

        verts, faces_t, norms = surface.get_regular_mesh(num_u_segments, num_v_segments)
        verts_np = verts.numpy()
        norms_np = norms.numpy()
        faces_np_t = faces_t.numpy()

        # update mesh vertices
        mesh.set_verts(verts_np[faces_np_t])

        # update face colors using UVs
        uvs = surface.get_regular_uv(num_u_segments, num_v_segments)
        uvs = (uvs - uvs.amin(0)) / (uvs.amax(0) - uvs.amin(0))
        uvs_np_t = uvs.numpy()
        colors_t = np.concatenate([uvs_np_t, np.full_like(uvs_np_t[..., 0:1], 0.5)], axis=-1)
        mesh.set_facecolor(colors_t[faces_np_t].mean(-2))

        # replace quiver with new normals (remove and redraw)
        nonlocal quiver_plot
        try:
            quiver_plot.remove()
        except Exception:
            pass
        quiver_plot = ax.quiver(*verts_np.T, *norms_np.T, color=COLORS["normal"], length=0.1, arrow_length_ratio=0.2)
        quiver_plot.set_visible(quiver_visible[0])

        ax.set_title(f"Superellipsoid (e1={surface.e1:.2f}, e2={surface.e2:.2f})", fontsize=14)
        plt.draw()

    slider_e1.on_changed(update)
    slider_e2.on_changed(update)

    ax.legend()
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=45)
    ax.set_title(f"Superellipsoid (e1={surface.e1:.2f}, e2={surface.e2:.2f})", fontsize=14)

    plt.show()


if __name__ == "__main__":
    num_u_segments, num_v_segments = 48, 24
    superell = SuperellipsoidSurface(e1=1.0, e2=1.0)
    visualize_superellipsoid(superell, num_u_segments, num_v_segments)
