from pathlib import Path
import sys

import matplotlib
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import seaborn as sns

sys.path.append(Path(__file__).resolve().parents[3].as_posix())

from src.curve import BezierCurve

sns.set_style(style="whitegrid")
COLORS = {
    "control": "#e74c3c",
    "curve": "#3498db",
    "tangent": "#9b59b6",
    "acceleration": "#2ecc71",
    "curvature": "#e67e22",
    "circle": "#1abc9c",
    "vertex": "#3498db",
}


def _plot_curve(
    ax: Axes,
    control_points: np.ndarray,
    vertices: np.ndarray,
    control_points_label: str = "Control Points",
    control_points_color: str = COLORS["control"],
    vertex_label: str = "Curve",
    vertex_color: str = COLORS["vertex"],
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


def _plot_curve_with_control_points(ax: Axes, control_points: np.ndarray, vertices: np.ndarray, dim: int) -> None:
    _plot_curve(
        ax,
        control_points,
        vertices,
        control_points_label="Control Points",
        control_points_color=COLORS["control"],
        vertex_label="Curve",
        vertex_color=COLORS["vertex"],
    )

    if dim == 2:
        ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.75)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    ax.set_title(f"{dim}D Bezier Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    if dim == 3:
        ax.set_zlabel("Z", fontsize=12)
        ax.view_init(elev=45, azim=-135)


def _plot_vectors(ax: Axes, vertices: np.ndarray, vectors: np.ndarray, title: str, dim: int, degree: int) -> None:
    color = COLORS["tangent"] if "Tangent" in title else COLORS["acceleration"]

    ax.plot(*vertices.T, "-", color=COLORS["curve"], alpha=0.75, linewidth=2, label="Curve")
    if dim == 2:
        ax.quiver(
            *vertices.T,
            *vectors.T,
            color=color,
            scale=1 / degree,
            scale_units="xy",
            angles="xy",
            width=0.005,
            headwidth=3,
            headlength=4,
            headaxislength=3,
            alpha=0.75,
            label="Vectors"
        )
    else:
        ax.quiver(*vertices.T, *vectors.T, color=color, arrow_length_ratio=0.2, alpha=0.8, linewidth=1.5, label="Vectors")

    if dim == 2:
        ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.75)
    ax.legend(loc="best", frameon=True)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    if dim == 3:
        ax.set_zlabel("Z", fontsize=12)
        ax.view_init(elev=45, azim=-135)


def _plot_curvature(ax: Axes, t_values: np.ndarray, curvatures: np.ndarray) -> None:
    ax.plot(t_values, curvatures, "-", color=COLORS["curvature"], linewidth=2.5, alpha=0.9)

    ax.grid(True, linestyle="--", alpha=0.75)
    ax.set_title("Curvature Along Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Parameter t", fontsize=12)
    ax.set_ylabel("Curvature", fontsize=12)
    ax.set_ylim(bottom=0)


def _plot_osculating_circles(
    ax: Axes, vertices: np.ndarray, tangents: np.ndarray, accelerations: np.ndarray, curvatures: np.ndarray
) -> None:
    ax.plot(vertices[:, 0], vertices[:, 1], "-", color=COLORS["curve"], alpha=0.4, linewidth=3, label="Curve")

    step = max(1, len(vertices) // 5)
    for i in range(step, len(vertices) - step, step):
        if abs(curvatures[i]) > 1e-5:
            radius = 1 / curvatures[i]
            tangent = tangents[i]
            acceleration = accelerations[i]
            normal = np.array([-tangent[1], tangent[0]])
            normal /= np.linalg.norm(normal)

            sign = np.sign(tangent[0] * acceleration[1] - tangent[1] * acceleration[0])
            center = vertices[i] + sign * radius * normal

            circle = plt.Circle(center, abs(radius), color=COLORS["circle"], fill=False, alpha=0.8, linewidth=1, linestyle="-")
            ax.add_artist(circle)
            ax.plot([vertices[i][0], center[0]], [vertices[i][1], center[1]], "--", color=COLORS["circle"], alpha=0.75, linewidth=1.5)

    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.75)
    ax.legend(loc="best", frameon=True)
    ax.set_title("Osculating Circles", fontsize=14, fontweight="bold")


def visualize_bezier_curve(curve: BezierCurve, num_segments: int = 100, figsize: tuple[int, int] = (16, 8)) -> None:
    if num_segments < 2:
        raise ValueError("num_segments must be at least 2 for meaningful visualization.")

    control_point_np = curve.control_point.numpy()
    t_np = curve.get_regular_t(num_segments).numpy()
    vertices_np = curve.get_regular_vertex(num_segments).numpy()
    tangents_np = curve.get_regular_tangent(num_segments).numpy()
    accelerations_np = curve.get_regular_acceleration(num_segments).numpy()
    curvatures_np = curve.get_regular_curvature(num_segments).numpy()

    dim = curve.dimension
    mosaic = [["main", "main", "tangent", "acceleration"], ["main", "main", "curvature", "osculating"]]

    fig = plt.figure(figsize=figsize)
    axes = fig.subplot_mosaic(mosaic)
    if dim == 3:
        for key in ["main", "tangent", "acceleration"]:
            axes[key].remove()
            axes[key] = fig.add_subplot(axes[key].get_subplotspec(), projection="3d", elev=45, azim=-135)

    fig.suptitle(f"{dim}D Bezier Curve Visualization", fontsize=18, fontweight="bold", y=0.98)

    _plot_curve_with_control_points(axes["main"], control_point_np, vertices_np, dim)
    _plot_vectors(axes["tangent"], vertices_np, tangents_np, "Tangent Vectors", dim, curve.degree)
    _plot_vectors(axes["acceleration"], vertices_np, accelerations_np, "Acceleration Vectors", dim, curve.degree)
    _plot_curvature(axes["curvature"], t_np, curvatures_np)

    if dim == 2:
        _plot_osculating_circles(axes["osculating"], vertices_np, tangents_np, accelerations_np, curvatures_np)
    else:
        fig.delaxes(axes["osculating"])

    plt.tight_layout()
    plt.show()


def visualization_split_bezier_curve(curve: BezierCurve, figsize: tuple[int, int] = (7, 7), num_segments: int = 100) -> None:
    if curve.dimension == 2:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d", elev=45, azim=-135)
    plt.subplots_adjust(bottom=0.25)

    init_t = 0.5
    curve_left, curve_right = curve.split(init_t)

    _plot_curve(
        ax,
        curve.control_point.numpy(),
        curve.get_regular_vertex(num_segments).numpy(),
        control_points_label="Original Control Points",
        control_points_color=COLORS["curve"],
        vertex_label="Original Curve",
        vertex_color=COLORS["curve"],
    )

    left_cp, left_line = _plot_curve(
        ax,
        curve_left.control_point.numpy(),
        curve_left.get_regular_vertex(num_segments).numpy(),
        control_points_label="Left Control Points",
        control_points_color="#e74c3c",
        vertex_label="Left Curve",
        vertex_color="#e74c3c",
    )

    right_cp, right_line = _plot_curve(
        ax,
        curve_right.control_point.numpy(),
        curve_right.get_regular_vertex(num_segments).numpy(),
        control_points_label="Right Control Points",
        control_points_color="#2ecc71",
        vertex_label="Right Curve",
        vertex_color="#2ecc71",
    )

    if curve.dimension == 2:
        ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    ax.set_title("Interactive Bezier Curve Splitting", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    if curve.dimension == 3:
        ax.set_zlabel("Z", fontsize=12)

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor="#f0f0f0")
    t_slider = Slider(
        ax=ax_slider,
        label="t",
        valmin=0.0,
        valmax=1.0,
        valinit=init_t,
        valstep=0.01,
        color=COLORS["curve"],
        track_color="#e0e0e0",
    )

    def update(val) -> None:
        t = t_slider.val
        curve_left, curve_right = curve.split(t)

        left_vertices = curve_left.get_regular_vertex(num_segments).numpy()
        left_line.set_data(*left_vertices.T[: 2])
        left_control_points = curve_left.control_point.numpy()
        left_cp.set_data(*left_control_points.T[: 2])

        right_vertices = curve_right.get_regular_vertex(num_segments).numpy()
        right_line.set_data(*right_vertices.T[: 2])
        right_control_points = curve_right.control_point.numpy()
        right_cp.set_data(*right_control_points.T[: 2])

        if curve.dimension == 3:
            left_line.set_3d_properties(left_vertices[:, 2])
            left_cp.set_3d_properties(left_control_points[:, 2])

            right_line.set_3d_properties(right_vertices[:, 2])
            right_cp.set_3d_properties(right_control_points[:, 2])

        fig.canvas.draw_idle()

    t_slider.on_changed(update)

    plt.show()


def visualization_degree_elevation(curve: "BezierCurve", figsize: tuple[int, int] = (7, 7), num_segments: int = 100) -> None:
    if curve.dimension == 2:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d", elev=45, azim=-135)
    plt.subplots_adjust(bottom=0.25)

    init_degree_increase = 0
    upgraded_curve = curve

    _plot_curve(
        ax,
        curve.control_point.numpy(),
        curve.get_regular_vertex(num_segments).numpy(),
        control_points_label="Original Control Points",
        control_points_color=COLORS["control"],
        vertex_label="Original Curve",
        vertex_color=COLORS["control"],
    )

    upgraded_cp, upgraded_curve_plot = _plot_curve(
        ax,
        upgraded_curve.control_point.numpy(),
        upgraded_curve.get_regular_vertex(num_segments).numpy(),
        control_points_label="Upgraded Control Points",
        control_points_color=COLORS["curve"],
        vertex_label="Upgraded Curve",
        vertex_color=COLORS["curve"],
    )

    if curve.dimension == 2:
        ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    ax.set_title(f"Bezier Curve Degree Elevation (Current Degree: {curve.degree})", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    if curve.dimension == 3:
        ax.set_zlabel("Z", fontsize=12)

    slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor="#f0f0f0")
    degree_slider = Slider(
        ax=slider_ax,
        label="Degree Increase",
        valmin=0,
        valmax=100,
        valinit=init_degree_increase,
        valstep=1,
        color=COLORS["curve"],
        track_color="#e0e0e0",
    )

    def update(val):
        degree_increase = int(degree_slider.val)

        current_curve = curve
        current_curve = current_curve.elevate(current_curve.degree + degree_increase)

        upgraded_cp_points = current_curve.control_point.numpy()
        upgraded_vertices = current_curve.get_regular_vertex(num_segments).numpy()

        upgraded_cp.set_data(*upgraded_cp_points.T[: 2])
        upgraded_curve_plot.set_data(*upgraded_vertices.T[: 2])
        if curve.dimension == 3:
            upgraded_cp.set_3d_properties(upgraded_cp_points[:, 2])
            upgraded_curve_plot.set_3d_properties(upgraded_vertices[:, 2])

        current_degree = curve.degree + degree_increase
        ax.set_title(f"Bezier Curve Degree Elevation (Current Degree: {current_degree})", fontsize=16, fontweight="bold", pad=15)

        fig.canvas.draw_idle()

    degree_slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    num_segments = 51

    control_points_2d = [[0, 0], [1, 2], [3, -1], [4, 1]]
    curve_2d = BezierCurve(control_points_2d)
    visualize_bezier_curve(curve_2d, num_segments=num_segments)
    visualization_split_bezier_curve(curve_2d)
    visualization_degree_elevation(curve_2d, num_segments=num_segments)

    control_points_3d = [[0, 0, 0], [1, 2, 1], [3, -1, 2], [4, 1, 1]]
    curve_3d = BezierCurve(control_points_3d)
    visualize_bezier_curve(curve_3d, num_segments=num_segments)
    visualization_split_bezier_curve(curve_3d)
    visualization_degree_elevation(curve_3d, num_segments=num_segments)
