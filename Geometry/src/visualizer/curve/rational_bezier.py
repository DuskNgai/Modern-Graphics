from pathlib import Path
import sys

import matplotlib
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
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
}


def _plot_curve(
    ax: Axes,
    control_points: np.ndarray,
    vertices: np.ndarray,
    control_points_label: str = "Control Points",
    control_points_color: str = COLORS["homogeneous"],
    vertex_label: str = "Curve ",
    vertex_color: str = COLORS["homogeneous"],
) -> tuple[matplotlib.lines.Line2D, matplotlib.lines.Line2D]:
    cp, = ax.plot(
        *control_points.T,
        "o-",
        color=control_points_color,
        alpha=0.75,
        linewidth=1,
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label=control_points_label
    )

    line, = ax.plot(
        *vertices.T,
        "-",
        color=vertex_color,
        alpha=0.75,
        linewidth=2.5,
        label=vertex_label,
    )
    return cp, line


def visualize_rational_bezier_curve(
    curve: BezierCurve, rational_curve: RationalBezierCurve, figsize: tuple = (11, 11), num_segments: int = 100
) -> None:
    if curve.dimension == 2:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d", elev=45, azim=-135)

    fig.suptitle(f"{rational_curve.dimension}D Rational Bezier Curve Visualization", fontsize=16)

    _plot_curve(
        ax,
        curve.control_point.numpy(),
        curve.get_regular_vertex(num_segments).numpy(),
        control_points_label="Control Points (Homogeneous)",
        control_points_color=COLORS["homogeneous"],
        vertex_label="Curve (Homogeneous)",
        vertex_color=COLORS["homogeneous"],
    )

    _plot_curve(
        ax,
        rational_curve.control_point.numpy(),
        rational_curve.get_regular_vertex(num_segments).numpy(),
        control_points_label="Control Points (Rational)",
        control_points_color=COLORS["rational"],
        vertex_label="Curve (Rational)",
        vertex_color=COLORS["rational"],
    )

    origin = np.array([0, 0, 0])
    for control_point in curve.control_point:
        line_points = np.vstack([origin, control_point])
        ax.plot(*line_points.T, 'k:', alpha=0.3, linewidth=1)
    ax.scatter(*origin, c='k', s=40, label="Origin")

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
    num_segments = 51

    control_points_2d = [[0, 1, 1], [1, 1, 1], [2, 0, 2]]
    curve_2d = BezierCurve(control_points_2d)
    rational_curve_2d = RationalBezierCurve(control_points_2d)
    visualize_rational_bezier_curve(curve_2d, rational_curve_2d, num_segments=num_segments)
