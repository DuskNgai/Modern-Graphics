import torch

from .curve import ParametricCurve


class BezierCurve(ParametricCurve):

    def __init__(self, control_point: torch.Tensor) -> None:
        self.control_point = torch.as_tensor(control_point) # [n + 1, d]
        self.degree = self.control_point.shape[0] - 1

    def get_regular_vertex(self, num_segments: int) -> torch.Tensor:
        t = self.get_regular_t(num_segments)
        return self.evaluate(t)

    def get_regular_tangent(self, num_segments: int) -> torch.Tensor:
        t = self.get_regular_t(num_segments)
        return self.evaluate_tangent(t)

    def get_regular_acceleration(self, num_segments: int) -> torch.Tensor:
        t = self.get_regular_t(num_segments)
        return self.evaluate_acceleration(t)

    def get_regular_curvature(self, num_segments: int) -> torch.Tensor:
        t = self.get_regular_t(num_segments)
        tangent = self.evaluate_tangent(t)
        acceleration = self.evaluate_acceleration(t)
        return acceleration / ((1 + tangent ** 2) ** 3).sqrt()

    def evaluate(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bezier curve at parameter t using the Bernstein basis matrix form.

        Args:
            t (`torch.Tensor`): Shape [..., 1] or [...], values in [0, 1]

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        return self.evaluate_recursive(t, 0, self.degree)

    def evaluate_tangent(self, t: torch.Tensor) -> torch.Tensor:
        """
        n * {P[1:n](t) - P[0:n - 1](t)}

        Args:
            t (`torch.Tensor`): Shape [..., 1]

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        return self.degree * (self.evaluate_recursive(t, 1, self.degree) - self.evaluate_recursive(t, 0, self.degree - 1))

    def evaluate_acceleration(self, t: torch.Tensor) -> torch.Tensor:
        """
        n * (n - 1) * {P[2:n](t) - 2 * P[1:n - 1](t) + P[0:n - 2](t)}

        Args:
            t (`torch.Tensor`): Shape [..., 1]

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        return self.degree * (self.degree - 1) * (
            self.evaluate_recursive(t, 2, self.degree) - \
            self.evaluate_recursive(t, 1, self.degree - 1) * 2 + \
            self.evaluate_recursive(t, 0, self.degree - 2)
        )

    def evaluate_recursive(self, t: torch.Tensor, start_index: int, end_index: int) -> torch.Tensor:
        """
        (1 - t) * P[s: e - 1] + t * P[s + 1: e]
        """
        if start_index == end_index:
            return self.control_point[start_index]
        else:
            return (1 - t) * self.evaluate_recursive(t, start_index, end_index - 1) + \
                         t * self.evaluate_recursive(t, start_index + 1, end_index)

    def split(self, t: float) -> tuple["BezierCurve", "BezierCurve"]:
        t_tensor = torch.tensor(t, dtype=self.control_point.dtype, device=self.control_point.device)
        control_point_left = torch.stack([self.evaluate_recursive(t_tensor, 0, i) for i in range(self.degree + 1)])
        control_point_right = torch.stack([self.evaluate_recursive(t_tensor, i, self.degree) for i in range(self.degree + 1)])
        return BezierCurve(control_point_left), BezierCurve(control_point_right)
