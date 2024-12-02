import numpy as np

from curve import ParametricCurve

class BezierCurve(ParametricCurve):
    def __init__(self,
        order: int,
        control_point: np.ndarray
    ) -> None:
        self.verify_shape_of_control_point(order, control_point)
        self.order = order
        self.control_point = control_point

    def verify_shape_of_control_point(self, order: int, control_point: np.ndarray) -> None:
        assert control_point.shape[-2] == (order + 1)

    @classmethod
    def generate_regular_t(cls, n: int) -> np.ndarray:
        """
        0 <= t <= 1

        Return:
            (`np.ndarray`): Shape [n, 1]
        """
        return np.linspace(0.0, 1.0, n)[..., np.newaxis]

    def generate_regular_vertex(self, num_segments: int) -> np.ndarray:
        """
        Return:
            (`np.ndarray`): Shape [m, d]
        """
        t = self.generate_regular_t(num_segments)
        return self.evaluate(t)

    def generate_regular_tangent(self, num_segments: int) -> np.ndarray:
        """
        Return:
            (`np.ndarray`): Shape [m, d]
        """
        t = self.generate_regular_t(num_segments)
        return self.evaluate_tangent(t)

    def generate_regular_acceleration(self, num_segments: int) -> np.ndarray:
        t = self.generate_regular_t(num_segments)
        return self.evaluate_acceleration(t)

    def generate_regular_curvature(self, num_segments: int) -> np.ndarray:
        t = self.generate_regular_t(num_segments)
        tangent = self.evaluate_tangent(t)
        acceleration = self.evaluate_acceleration(t)
        return acceleration / (1 + tangent ** 2) ** (3 / 2)

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """
        (1 - t) * P[0:n - 1](t) + t * P[1:n](t)

        Args:
            t (`np.ndarray`): Shape [..., 1]

        Return:
            (`np.ndarray`): Shape [..., d]
        """
        return self.evaluate_recursive(t, 0, self.order)

    def evaluate_tangent(self, t: np.ndarray) -> np.ndarray:
        """
        n * {P[1:n](t) - P[0:n - 1](t)}

        Args:
            t (`np.ndarray`): Shape [..., 1]

        Return:
            (`np.ndarray`): Shape [..., d]
        """
        return self.order * (self.evaluate_recursive(t, 1, self.order) - self.evaluate_recursive(t, 0, self.order - 1))

    def evaluate_acceleration(self, t: np.ndarray) -> np.ndarray:
        """
        n * (n - 1) * {P[2:n](t) - 2 * P[1:n - 1](t) + P[0:n - 2](t)}

        Args:
            t (`np.ndarray`): Shape [..., 1]

        Return:
            (`np.ndarray`): Shape [..., d]
        """
        return self.order * (self.order - 1) * (self.evaluate_recursive(t, 2, self.order) - 2 * self.evaluate_recursive(t, 1, self.order - 1) + self.evaluate_recursive(t, 0, self.order - 2))

    def evaluate_recursive(self, t: np.ndarray, start: int, end: int) -> np.ndarray:
        """
        (1 - t) * P[s: e - 1] + t * P[s + 1: e]
        """
        if start == end:
            return self.control_point[start]
        else:
            return (1 - t) * self.evaluate_recursive(t, start, end - 1) + t * self.evaluate_recursive(t, start + 1, end)

    def split(self, t: float) -> tuple["BezierCurve", "BezierCurve"]:
        control_point_left = np.array([
            self.evaluate_recursive(t, 0, i) for i in range(self.order + 1)
        ])
        control_point_right = np.array([
            self.evaluate_recursive(t, i, self.order) for i in range(self.order + 1)
        ])
        return BezierCurve(self.order, control_point_left), BezierCurve(self.order, control_point_right)
