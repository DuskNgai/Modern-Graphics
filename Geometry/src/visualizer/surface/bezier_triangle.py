from pathlib import Path
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns

sys.path.append(Path(__file__).resolve().parents[3].as_posix())

from src.surface import BezierTriangleSurface

sns.set_style(style="whitegrid")
COLORS = {
    "control": "#e74c3c",
    "surface": "#3498db",
    "normal": "#9b59b6",
}


def visualize_bezier_triangle(
    surface: BezierTriangleSurface,
    num_segments_per_edge: int = 10,
    normal_scale: float = 0.1,
    figsize: tuple = (10, 8),
) -> None:
    control_points = surface.control_point.numpy()
    vertex, face, normal, uvw = surface.get_regular_mesh(num_segments_per_edge)
    vertices = vertex.numpy()
    faces = face.numpy()
    normals = normal.numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(vertices[faces], alpha=0.5, edgecolor="none", facecolor=COLORS["surface"])
    ax.add_collection3d(mesh)

    ax.scatter(*control_points.T, color=COLORS["control"], s=50, label="Control Points")

    edges = []
    for i in range(surface.degree):
        l, r = i * (i + 1) // 2, (i + 1) * (i + 2) // 2 # noqa: E741
        for j in range(l, r):
            edges.append([j, j + i + 1])
            edges.append([j + i + 1, j + i + 2])
            edges.append([j + i + 2, j])

    for edge in edges:
        x = [control_points[edge[0], 0], control_points[edge[1], 0]]
        y = [control_points[edge[0], 1], control_points[edge[1], 1]]
        z = [control_points[edge[0], 2], control_points[edge[1], 2]]
        ax.plot(x, y, z, color=COLORS["control"], linewidth=1, alpha=0.75)

    ax.quiver(
        *vertices.T,
        *normals.T,
        color=COLORS["normal"],
        length=normal_scale,
        arrow_length_ratio=0.2,
        label="Normals",
    )

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=45, azim=-135)

    ax.set_title(f"Bezier Triangle Surface (Degree {surface.degree})", fontsize=14)

    plt.show()


if __name__ == "__main__":
    degree = 2
    control_points = [[1, 1, 1], [1, 0, 0.5], [0, 1, 0.5], [0.5, 0, 0], [0.0, 0.0, 0], [0, 0.5, 0]]
    surface = BezierTriangleSurface(degree, control_points)
    visualize_bezier_triangle(surface, num_segments_per_edge=10, normal_scale=0.1)
    # visualize_bezier_triangle_interactive(surface, num_segments_per_edge=10)
