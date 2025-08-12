from functools import lru_cache

import torch

from .bezier import BezierCurve

__all__ = ["RationalBezierCurve"]


class RationalBezierCurve(BezierCurve):

    def __init__(self, control_point: torch.Tensor) -> None:
        """
        Args:
            `control_point` (`torch.Tensor`): Shape `[n + 1, d + 1]`,
                where `n` is the degree of the curve,
                and `d` is the dimension of the curve.
        """
        super().__init__(control_point)
        self._dimension = self._control_point.shape[1] - 1

    @property
    @lru_cache(maxsize=1)
    def control_point(self) -> torch.Tensor:
        return self._control_point / self._control_point[..., -1 :]

    @lru_cache()
    def get_regular_vertex(self, num_segments: int) -> torch.Tensor:
        vertex = super().get_regular_vertex(num_segments)
        return vertex / vertex[..., -1 :]

    @lru_cache()
    def get_regular_tangent(self, num_segments: int) -> torch.Tensor:
        vertex = super().get_regular_vertex(num_segments)
        tangent = super().get_regular_tangent(num_segments)
        return (tangent - tangent[..., -1 :] * vertex / vertex[..., -1 :]) / vertex[..., -1 :]

    @lru_cache()
    def get_regular_acceleration(self, num_segments: int) -> torch.Tensor:
        raise NotImplementedError
