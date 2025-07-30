from curve.bezier import BezierCurve
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from visualize.visualizer import Visualizer

__all__ = ["BezierVisualizer"]


class BezierVisualizer(Visualizer):

    def __init__(self, curve: BezierCurve, num_segments: int) -> None:
        self.curve = curve
        self.num_segments = num_segments
        self._validate_num_segments()

        # Prepare data
        self.control_point_np = self.curve.control_point.numpy()
        self.t_np = self.curve.get_regular_t(self.num_segments).numpy()
        self.vertices_np = self.curve.get_regular_vertex(self.num_segments).numpy()
        self.tangents_np = self.curve.get_regular_tangent(self.num_segments).numpy()
        self.accelerations_np = self.curve.get_regular_acceleration(self.num_segments).numpy()
        self.curvatures_np = self.curve.get_regular_curvature(self.num_segments).numpy()

        # Normalize vectors for better visualization
        self.degree = self.curve.degree
        self.norm_tangents = self.tangents_np / self.degree
        self.norm_accelerations = self.accelerations_np / (self.degree * (self.degree - 1)) if self.degree > 1 else self.accelerations_np

    def _validate_num_segments(self) -> None:
        if self.num_segments < 2:
            raise ValueError("num_segments must be at least 2 for meaningful visualization.")

    def _plot_curve(self, ax: Axes) -> None:
        ax.plot(*self.control_point_np.T, "ro--", alpha=0.5, label="Control Polygon")
        ax.scatter(*self.vertices_np.T, c=self.t_np, cmap="viridis", s=40, label="Vertices (t)")
        ax.plot(*self.vertices_np.T, "b-", alpha=0.3, linewidth=5, label="Curve")

        ax.set_title(f"{self.curve.dimension}D Bezier Curve")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if self.curve.dimension == 3:
            ax.set_zlabel("Z")
        ax.grid(True)
        ax.legend()

    def _plot_vectors(self, ax: Axes, vectors: np.ndarray, title: str) -> None:
        if self.curve.dimension == 3:
            ax.quiver(*self.vertices_np.T, *vectors.T, color="#FF8080", arrow_length_ratio=0.2)
        else:
            ax.quiver(*self.vertices_np.T, *vectors.T, color="#FF8080", scale=1, scale_units="xy", angles="xy")
        ax.plot(*self.vertices_np.T, "b-", alpha=0.3, linewidth=5, label="Curve")

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if self.curve.dimension == 3:
            ax.set_zlabel("Z")
        ax.grid(True)
        ax.legend()

    def _plot_curvature_graph(self, ax: Axes) -> None:
        ax.plot(self.t_np, self.curvatures_np, "m-", linewidth=2)

        ax.set_title("Curvature Along Curve")
        ax.set_xlabel("Parameter t")
        ax.set_ylabel("Curvature")
        ax.grid(True)

    def _plot_osculating_circles(self, ax: Axes) -> None:
        ax.plot(*self.vertices_np.T, "b-", alpha=0.3, linewidth=5, label="Curve")

        step = max(1, len(self.vertices_np) // 5)
        for i in range(step, len(self.vertices_np) - step, step):
            if abs(self.curvatures_np[i]) > 1e-5:
                radius = 1 / self.curvatures_np[i]
                tangent = self.tangents_np[i]
                normal = np.array([-tangent[1], tangent[0]])
                normal /= np.linalg.norm(normal)

                sign = np.sign(tangent[0] * self.accelerations_np[i][1] - tangent[1] * self.accelerations_np[i][0])
                center = self.vertices_np[i] + sign * radius * normal

                circle = plt.Circle(center, abs(radius), color="g", fill=False, alpha=0.7)
                ax.add_artist(circle)
                ax.plot([self.vertices_np[i][0], center[0]], [self.vertices_np[i][1], center[1]], "g--")

        ax.set_title("Osculating Circles")
        ax.axis("equal")
        ax.grid(True)

    def visualize(self, figsize: tuple = (21, 11)) -> None:
        mosaic = [
            ["main", "main", "tangent", "acceleration"],
            ["main", "main", "curvature", "osculating"],
        ]

        fig, axes = plt.subplot_mosaic(mosaic, figsize=figsize)
        fig.suptitle(f"{self.curve.dimension}D Bezier Curve Visualization", fontsize=16)

        for key in ["main", "tangent", "acceleration"]:
            if self.curve.dimension == 3:
                axes[key].remove()
                axes[key] = fig.add_subplot(axes[key].get_subplotspec(), projection="3d", azim=-45, elev=30)

        self._plot_curve(axes["main"])
        self._plot_vectors(axes["tangent"], self.norm_tangents, "Tangent Vectors")
        self._plot_vectors(axes["acceleration"], self.norm_accelerations, "Acceleration Vectors")
        self._plot_curvature_graph(axes["curvature"])

        if self.curve.dimension == 2:
            self._plot_osculating_circles(axes["osculating"])
        else:
            fig.delaxes(axes["osculating"])

        plt.tight_layout()
        plt.show()
