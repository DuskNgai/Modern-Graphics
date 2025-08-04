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
        self._control_point = torch.as_tensor(control_point, dtype=torch.float32) # [n + 1, d]
        self._degree = self._control_point.shape[0] - 1
        self._dimension = self._control_point.shape[1]

    @property
    @lru_cache(maxsize=1)
    def control_point(self) -> torch.Tensor:
        return self._control_point

    @property
    @lru_cache(maxsize=1)
    def degree(self) -> int:
        return self._degree

    @property
    @lru_cache(maxsize=1)
    def dimension(self) -> int:
        return self._dimension

    @lru_cache(maxsize=None)
    def get_regular_vertex(self, num_segments: int) -> torch.Tensor:
        return self.evaluate(self.get_regular_t(num_segments))

    @lru_cache(maxsize=None)
    def get_regular_tangent(self, num_segments: int) -> torch.Tensor:
        return self._evaluate_tangent(self.get_regular_t(num_segments))

    @lru_cache(maxsize=None)
    def get_regular_acceleration(self, num_segments: int) -> torch.Tensor:
        return self._evaluate_acceleration(self.get_regular_t(num_segments))

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
            `t` (`torch.Tensor`): Shape [...], values in [0, 1]

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        return self._evaluate_recurrent(t, 0, self.degree)

    def _evaluate_recursive(self, t: torch.Tensor, s: int, e: int) -> torch.Tensor:
        """
        C(t) = (1 - t) * P[s : e - 1] + t * P[s + 1 : e]

        Args:
            `t` (`torch.Tensor`): Shape [...], values in [0, 1]

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        if s == e:
            return self._control_point[s : e + 1]      # [1, d]
        else:
            return torch.lerp(
                self._evaluate_recursive(t, s, e - 1),
                self._evaluate_recursive(t, s + 1, e),
                t.unsqueeze(-1),
            )                                          # [..., d]

    def _evaluate_recurrent(self, t: torch.Tensor, s: int, e: int) -> torch.Tensor:
        """
        Recurrently evaluate Bezier curve at parameter t, which is more efficient than `_evaluate_recursive` for large `t`.
        """
        t = t.unsqueeze(-1)
        points = self._control_point[s : e + 1].clone().unsqueeze(-2) # [e - s + 1, d]
        for _ in range(e - s):
            points = torch.lerp(points[:-1], points[1 :], t)          # [e - s, ..., d]
        return points.squeeze(-3)                                     # [..., d]

    def _evaluate_tangent(self, t: torch.Tensor) -> torch.Tensor:
        """
        C'(t) = n * {P[1 : n](t) - P[0 : n - 1](t)}

        Args:
            `t` (`torch.Tensor`): Shape [...], values in [0, 1]

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        return self.degree * (
            self._evaluate_recurrent(t, 1, self.degree) - \
            self._evaluate_recurrent(t, 0, self.degree - 1)
        )

    def _evaluate_acceleration(self, t: torch.Tensor) -> torch.Tensor:
        """
        C''(t) = n * (n - 1) * {P[2 : n](t) - 2 * P[1:n - 1](t) + P[0 : n - 2](t)}

        Args:
            `t` (`torch.Tensor`): Shape [...], values in [0, 1]

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        return self.degree * (self.degree - 1) * (
            self._evaluate_recurrent(t, 2, self.degree) - \
            self._evaluate_recurrent(t, 1, self.degree - 1) * 2 + \
            self._evaluate_recurrent(t, 0, self.degree - 2)
        )

    def split(self, t: float) -> tuple["BezierCurve", "BezierCurve"]:
        """
        Split Bezier curve at parameter t.

        Args:
            `t` (`float`): Value in [0, 1]

        Return:
            (`BezierCurve`, `BezierCurve`): Two Bezier curves.
        """
        t_tensor = torch.tensor(t, dtype=self.control_point.dtype, device=self.control_point.device)
        control_point_left = torch.cat([self._evaluate_recurrent(t_tensor, 0, i) for i in range(self.degree + 1)])
        control_point_right = torch.cat([self._evaluate_recurrent(t_tensor, i, self.degree) for i in range(self.degree + 1)])
        return BezierCurve(control_point_left), BezierCurve(control_point_right)

    def elevate(self, new_degree: int) -> "BezierCurve":
        """
        Elevate the degree of the Bezier curve to `new_degree`.

        Args:
            `new_degree` (`int`): The new degree of the Bezier curve.

        Return:
            (`BezierCurve`): The elevated Bezier curve.
        """
        if new_degree < self.degree:
            raise ValueError("`new_degree` must be >= current degree")
        if new_degree == self.degree:
            return BezierCurve(self.control_point.clone())

        cp = self.control_point
        for deg in range(self.degree, new_degree):
            alpha = torch.arange(1, deg + 1, device=cp.device, dtype=cp.dtype).unsqueeze(1) / (deg + 1)
            cp = torch.cat([cp[: 1], torch.lerp(cp[1 :], cp[:-1], alpha), cp[-1 :]], dim=0)
        return BezierCurve(cp)
