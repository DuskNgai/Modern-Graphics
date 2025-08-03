from functools import lru_cache

import torch

from .curve import ParametricCurve

__all__ = ["BezierCurve"]


class BezierCurve(ParametricCurve):

    def __init__(self, control_point: torch.Tensor) -> None:
        """
        Args:
            `control_point` (`torch.Tensor`): Shape `[n + 1, d]`,
                where `n` is the degree of the curve,
                and `d` is the dimension of the curve.
        """
        self.control_point = torch.as_tensor(control_point, dtype=torch.float32) # [n + 1, d]

    @property
    def degree(self) -> int:
        return self.control_point.shape[0] - 1

    @property
    def dimension(self) -> int:
        return self.control_point.shape[1]

    @lru_cache(maxsize=None)
    def get_regular_vertex(self, num_segments: int) -> torch.Tensor:
        t = self.get_regular_t(num_segments)
        return self.evaluate(t)

    @lru_cache(maxsize=None)
    def get_regular_tangent(self, num_segments: int) -> torch.Tensor:
        t = self.get_regular_t(num_segments)
        return self._evaluate_tangent(t)

    @lru_cache(maxsize=None)
    def get_regular_acceleration(self, num_segments: int) -> torch.Tensor:
        t = self.get_regular_t(num_segments)
        return self._evaluate_acceleration(t)

    @lru_cache(maxsize=None)
    def get_regular_curvature(self, num_segments: int) -> torch.Tensor:
        tangent = self.get_regular_tangent(num_segments)
        acceleration = self.get_regular_acceleration(num_segments)

        denominator = tangent.norm(dim=-1).pow(3) + 1e-8

        if self.dimension == 2:
            return torch.stack([tangent, acceleration], dim=-1).det().abs() / denominator
        elif self.dimension == 3:
            return torch.cross(tangent, acceleration, dim=-1).norm(dim=-1) / denominator
        else:
            raise ValueError("Curvature is only defined in 2D and 3D.")

    def evaluate(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bezier curve at parameter t.

        Args:
            `t` (`torch.Tensor`): Shape [..., 1], values in [0, 1]

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        return self._evaluate_recursive(t, 0, self.degree)

    def _evaluate_recursive(self, t: torch.Tensor, start_index: int, end_index: int) -> torch.Tensor:
        """
        C(t) = (1 - t) * P[s : e - 1] + t * P[s + 1 : e]

        Args:
            `t` (`torch.Tensor`): Shape [..., 1], values in [0, 1]

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        if start_index == end_index:
            return self.control_point[start_index]
        else:
            return self._evaluate_recursive(t, start_index, end_index - 1) * (1 - t) + \
                   self._evaluate_recursive(t, start_index + 1, end_index) * t

    def _evaluate_tangent(self, t: torch.Tensor) -> torch.Tensor:
        """
        C'(t) = n * {P[1 : n](t) - P[0 : n - 1](t)}

        Args:
            `t` (`torch.Tensor`): Shape [..., 1], values in [0, 1]

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        return self.degree * (
            self._evaluate_recursive(t, 1, self.degree) - \
            self._evaluate_recursive(t, 0, self.degree - 1)
        )

    def _evaluate_acceleration(self, t: torch.Tensor) -> torch.Tensor:
        """
        C''(t) = n * (n - 1) * {P[2 : n](t) - 2 * P[1:n - 1](t) + P[0 : n - 2](t)}

        Args:
            `t` (`torch.Tensor`): Shape [..., 1], values in [0, 1]

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        return self.degree * (self.degree - 1) * (
            self._evaluate_recursive(t, 2, self.degree) - \
            self._evaluate_recursive(t, 1, self.degree - 1) * 2 + \
            self._evaluate_recursive(t, 0, self.degree - 2)
        )

    def split(self, t: float) -> tuple["BezierCurve", "BezierCurve"]:
        t_tensor = torch.tensor(t, dtype=self.control_point.dtype, device=self.control_point.device)
        control_point_left = torch.stack([self._evaluate_recursive(t_tensor, 0, i) for i in range(self.degree + 1)])
        control_point_right = torch.stack([self._evaluate_recursive(t_tensor, i, self.degree) for i in range(self.degree + 1)])
        return BezierCurve(control_point_left), BezierCurve(control_point_right)
