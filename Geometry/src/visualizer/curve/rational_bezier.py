from pathlib import Path
import sys

import matplotlib
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import seaborn as sns

sys.path.append(Path(__file__).resolve().parents[3].as_posix())

from src.curve import (
    BezierCurve,
    RationalBezierCurve,
)

sns.set_style(style="whitegrid")
COLORS = {
    "homogeneous": "#e74c3c",
    "rational": "#3498db",
    "tangent": "#FF8080",
    "acceleration": "#9b59b6",
}


def _plot_curve(
    ax: Axes,
    vertices: np.ndarray,
    color: np.ndarray | str,
    label: str = "Curve",
) -> LineCollection:
    lc = Line3DCollection(
        np.stack([vertices[:-1], vertices[1 :]], axis=1),
        cmap="turbo",
        norm=matplotlib.colors.Normalize(0, 1),
    )
    lc.set_alpha(0.75)
    if isinstance(color, np.ndarray):
        lc.set_array(color)
    else:
        lc.set_color(color)
    lc.set_label(label)
    lc.set_linewidth(2.5)
    ax.add_collection(lc)

    return lc


def _plot_control_points(
    ax: Axes,
    control_points: np.ndarray,
    color: str = COLORS["rational"],
    label: str = "Control Points",
) -> matplotlib.lines.Line2D:
    cp, = ax.plot(
        *control_points.T,
        "o-",
        color=color,
        alpha=0.75,
        linewidth=1,
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label=label,
    )
    return cp


def _plot_vectors(ax: Axes, vertices: np.ndarray, vectors: np.ndarray, color: str, label: str) -> matplotlib.quiver.Quiver:
    vectors = ax.quiver(*vertices.T, *vectors.T, color=color, arrow_length_ratio=0.1, alpha=0.8, linewidth=1.5, label=label)

    ax.grid(True, linestyle="--", alpha=0.75)
    ax.legend(loc="best", frameon=True)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    ax.view_init(elev=45, azim=-135)

    return vectors


def visualize_rational_bezier_curve(
    curve: BezierCurve, rational_curve: RationalBezierCurve, figsize: tuple = (11, 11), num_segments: int = 100
) -> None:
    if num_segments < 2:
        raise ValueError("num_segments must be at least 2 for meaningful visualization.")
    t_np = curve.get_regular_t(num_segments).numpy()

    control_point_homo_np = curve.control_point.numpy()
    vertices_homo_np = curve.get_regular_vertex(num_segments).numpy()
    tangents_homo_np = curve.get_regular_tangent(num_segments).numpy()

    control_point_rational_np = rational_curve.control_point.numpy()
    vertices_rational_np = rational_curve.get_regular_vertex(num_segments).numpy()
    tangents_rational_np = rational_curve.get_regular_tangent(num_segments).numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d", elev=45, azim=-135)

    fig.suptitle(f"{rational_curve.dimension}D Rational Bezier Curve Visualization", fontsize=16)

    _plot_curve(ax, vertices_homo_np, color=t_np, label="Curve (Homogeneous)")
    _plot_control_points(ax, control_point_homo_np, color=COLORS["homogeneous"], label="Control Points (Homogeneous)")
    tangents_homo = _plot_vectors(
        ax, vertices_homo_np, tangents_homo_np, color=COLORS["homogeneous"], label="Tangent Vectors (Homogeneous)"
    )

    _plot_curve(ax, vertices_rational_np, color=t_np, label="Curve (Rational)")
    _plot_control_points(ax, control_point_rational_np, color=COLORS["rational"], label="Control Points (Rational)")
    tangents_rational = _plot_vectors(
        ax, vertices_rational_np, tangents_rational_np, color=COLORS["rational"], label="Tangent Vectors (Rational)"
    )

    origin = np.array([0, 0, 0])
    for control_point in curve.control_point:
        line_points = np.vstack([origin, control_point])
        ax.plot(*line_points.T, 'k:', alpha=0.3, linewidth=1)
    ax.scatter(*origin, c='k', s=40, label="Origin")

    tangents_homo.set_visible(False)
    tangents_rational.set_visible(False)

    ax_button = plt.axes([0.3, 0.05, 0.4, 0.06])
    toggle_button = Button(ax_button, "Toggle Tangents")

    def toggle_normals(event):
        visible = tangents_homo.get_visible()
        tangents_homo.set_visible(not visible)
        tangents_rational.set_visible(not visible)
        plt.draw()

    toggle_button.on_clicked(toggle_normals)

    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    ax.set_title(f"{rational_curve.dimension}D Bezier Curve")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    num_segments = 11

    control_points_2d = [[1, 0, 1], [1, 1, 1], [0, 2, 2]]
    curve_2d = BezierCurve(control_points_2d)
    rational_curve_2d = RationalBezierCurve(control_points_2d)
    visualize_rational_bezier_curve(curve_2d, rational_curve_2d, num_segments=num_segments)
