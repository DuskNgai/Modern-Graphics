from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from ..visualizer import Visualizer
from src.curve import (
    BezierCurve,
    RationalBezierCurve,
)

__all__ = ["RationalBezierCurveVisualizer"]


class RationalBezierCurveVisualizer(Visualizer):

    def __init__(self, curve: RationalBezierCurve, num_segments: int) -> None:
        self.curve = curve
        self.num_segments = num_segments
        self._validate_num_segments()

        self.curve_homo = BezierCurve(self.curve.control_point)

        # Prepare data
        self.t_np = self.curve.get_regular_t(self.num_segments).numpy()

        self.control_point_homo_np = self.curve_homo.control_point.numpy()
        self.vertices_homo_np = self.curve_homo.get_regular_vertex(self.num_segments).numpy()

        self.control_points_np = self.control_point_homo_np / self.control_point_homo_np[:, -1 :]
        self.vertices_np = self.curve.get_regular_vertex(self.num_segments).numpy()

    def _validate_num_segments(self) -> None:
        if self.num_segments < 2:
            raise ValueError("num_segments must be at least 2 for meaningful visualization.")

    def _plot_curve(self, ax: Axes) -> None:
        ax.plot(*self.control_point_homo_np.T, "ro--", alpha=0.5, label="Control Polygon (Homogeneous)")
        ax.plot(*self.vertices_homo_np.T, "r-", alpha=0.3, linewidth=5, label="Curve (Homogeneous)")

        ax.plot(*self.control_points_np.T, "bo--", alpha=0.5, label="Control Polygon (Projected)")
        ax.plot(*self.vertices_np.T, "b-", alpha=0.3, linewidth=5, label="Curve (Projected)")

        origin = np.array([0, 0, 0])
        for homo_point in self.control_point_homo_np[:, : 3]:
            line_points = np.vstack([origin, homo_point])
            ax.plot(*line_points.T, 'k:', alpha=0.3, linewidth=1)
        ax.scatter(*origin, c='k', s=40)

        ax.set_title(f"{self.curve.dimension}D Bezier Curve")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.axis("equal")
        ax.grid(True)
        ax.legend()

    def visualize(self, figsize: tuple = (11, 11)) -> None:
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"{self.curve.dimension}D Rational Bezier Curve Visualization", fontsize=16)

        ax = fig.add_subplot(111, projection='3d')
        self._plot_curve(ax)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    control_points_2d = [[0, 1, 1], [1, 1, 1], [2, 0, 2]]
    curve_2d = RationalBezierCurve(control_points_2d)
    visualizer_2d = RationalBezierCurveVisualizer(curve_2d, num_segments=51)
    visualizer_2d.visualize()
