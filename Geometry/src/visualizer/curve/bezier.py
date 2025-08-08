from pathlib import Path
import sys

import matplotlib
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import seaborn as sns

sys.path.append(Path(__file__).resolve().parents[3].as_posix())

from src.curve import BezierCurve

sns.set_style(style="whitegrid")
COLORS = {
    "control": "#e74c3c",
    "curve": "#3498db",
    "tangent": "#FF8080",
    "acceleration": "#9b59b6",
    "curvature": "#2ecc71",
}


def _plot_curve(
    ax: Axes,
    vertices: np.ndarray,
    color: np.ndarray | str,
    label: str = "Curve",
) -> LineCollection:
    if vertices.shape[-1] == 2:
        LineClass = LineCollection
    else:
        LineClass = Line3DCollection

    lc = LineClass(
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
    color: str = COLORS["control"],
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


def _plot_curve_with_control_points(ax: Axes, control_points: np.ndarray, t: np.ndarray, vertices: np.ndarray, dim: int) -> None:
    _plot_curve(ax, vertices, color=t, label="Curve")
    _plot_control_points(ax, control_points, color=COLORS["control"], label="Control Points")

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


def _plot_vectors(ax: Axes, t: np.ndarray, vertices: np.ndarray, vectors: np.ndarray, title: str, dim: int, degree: int) -> None:
    color = COLORS["tangent"] if "Tangent" in title else COLORS["acceleration"]

    _plot_curve(ax, vertices, color=t, label="Curve")
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
            label="Vectors",
        )
    else:
        ax.quiver(*vertices.T, *vectors.T, color=color, arrow_length_ratio=0.1, alpha=0.8, linewidth=1.5, label="Vectors")

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


def _plot_curvature(ax: Axes, t: np.ndarray, curvatures: np.ndarray) -> None:
    ax.plot(t, curvatures, "-", color=COLORS["curvature"], linewidth=2.5, alpha=0.9)

    ax.grid(True, linestyle="--", alpha=0.75)
    ax.set_title("Curvature Along Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Parameter t", fontsize=12)
    ax.set_ylabel("Curvature", fontsize=12)
    ax.set_ylim(bottom=0)


def _plot_osculating_circles(
    ax: Axes, t: np.ndarray, vertices: np.ndarray, tangents: np.ndarray, accelerations: np.ndarray, curvatures: np.ndarray
) -> None:
    _plot_curve(ax, vertices, color=t, label="Curve")

    step = max(1, len(vertices) // 5)
    for i in range(step, len(vertices) - step, step):
        if abs(curvatures[i]) > 1e-5:
            radius = 1 / curvatures[i]

            normal = np.array([-tangents[i, 1], tangents[i, 0]])
            normal /= np.linalg.norm(normal)

            sign = np.sign(tangents[i, 0] * accelerations[i, 1] - tangents[i, 1] * accelerations[i, 0])
            center = vertices[i] + sign * radius * normal

            circle = plt.Circle(center, radius, color=COLORS["curvature"], fill=False, alpha=0.8, linewidth=1, linestyle="-")
            ax.add_artist(circle)
            ax.plot([vertices[i, 0], center[0]], [vertices[i, 1], center[1]], "--", color=COLORS["curvature"], alpha=0.75, linewidth=1.5)

    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.75)
    ax.legend(loc="best", frameon=True)
    ax.set_title("Osculating Circles", fontsize=14, fontweight="bold")


def visualize_bezier_curve(curve: BezierCurve, num_segments: int = 100, figsize: tuple[int, int] = (16, 8)) -> None:
    if num_segments < 2:
        raise ValueError("num_segments must be at least 2 for meaningful visualization.")
    t_np = curve.get_regular_t(num_segments).numpy()

    control_point_np = curve.control_point.numpy()
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

    _plot_curve_with_control_points(axes["main"], control_point_np, t_np, vertices_np, dim)
    _plot_vectors(axes["tangent"], t_np, vertices_np, tangents_np, "Tangent Vectors", dim, curve.degree)
    _plot_vectors(axes["acceleration"], t_np, vertices_np, accelerations_np, "Acceleration Vectors", dim, curve.degree)
    _plot_curvature(axes["curvature"], t_np, curvatures_np)

    if dim == 2:
        _plot_osculating_circles(axes["osculating"], t_np, vertices_np, tangents_np, accelerations_np, curvatures_np)
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
    t_np = curve.get_regular_t(num_segments).numpy()
    curve_left, curve_right = curve.split(init_t)

    _plot_curve(ax, curve.get_regular_vertex(num_segments).numpy(), color=t_np, label="Original Curve")
    _plot_control_points(ax, curve.control_point.numpy(), color=COLORS["control"], label="Original Control Points")

    left_line = _plot_curve(ax, curve_left.get_regular_vertex(num_segments).numpy(), color=COLORS["curve"], label="Left Curve")
    left_cp = _plot_control_points(ax, curve_left.control_point.numpy(), color=COLORS["curve"], label="Left Control Points")

    right_line = _plot_curve(ax, curve_right.get_regular_vertex(num_segments).numpy(), color=COLORS["curvature"], label="Right Curve")
    right_cp = _plot_control_points(ax, curve_right.control_point.numpy(), color=COLORS["curvature"], label="Right Control Points")

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
        left_line.set_segments(np.stack([left_vertices[:-1], left_vertices[1 :]], axis=1))
        left_control_points = curve_left.control_point.numpy()
        left_cp.set_data(*left_control_points.T[: 2])

        right_vertices = curve_right.get_regular_vertex(num_segments).numpy()
        right_line.set_segments(np.stack([right_vertices[:-1], right_vertices[1 :]], axis=1))
        right_control_points = curve_right.control_point.numpy()
        right_cp.set_data(*right_control_points.T[: 2])

        if curve.dimension == 3:
            left_cp.set_3d_properties(left_control_points[:, 2])
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
    t_np = curve.get_regular_t(num_segments).numpy()

    _plot_curve(ax, curve.get_regular_vertex(num_segments).numpy(), color=t_np, label="Original Curve")
    _plot_control_points(ax, curve.control_point.numpy(), color=COLORS["control"], label="Original Control Points")
    elevated_cp = _plot_control_points(ax, curve.control_point.numpy(), color="#3498db", label="Elevated Control Points")

    if curve.dimension == 2:
        ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    ax.set_title(f"Bezier Curve Degree Elevation (Current Degree: {curve.degree})", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    if curve.dimension == 3:
        ax.set_zlabel("Z", fontsize=12)

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor="#f0f0f0")
    degree_slider = Slider(
        ax=ax_slider,
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
        for _ in range(degree_increase):
            current_curve = current_curve.elevate()

        elevated_cp_points = current_curve.control_point.numpy()
        elevated_cp.set_data(*elevated_cp_points.T[: 2])
        if curve.dimension == 3:
            elevated_cp.set_3d_properties(elevated_cp_points[:, 2])

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
