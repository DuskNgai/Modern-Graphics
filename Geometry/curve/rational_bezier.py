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
        self.control_point = torch.as_tensor(control_point, dtype=torch.float32) # [n + 1, d + 1]

    @property
    def degree(self) -> int:
        return self.control_point.shape[0] - 1

    @property
    def dimension(self) -> int:
        return self.control_point.shape[1] - 1

    @lru_cache(maxsize=None)
    def get_regular_vertex(self, num_segments: int) -> torch.Tensor:
        vertex = super().get_regular_vertex(num_segments)
        return vertex / vertex[:, -1 :]
