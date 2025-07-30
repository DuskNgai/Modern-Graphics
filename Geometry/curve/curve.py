from abc import (
    ABCMeta,
    abstractmethod,
)
from functools import lru_cache

import torch


class ParametricCurve(metaclass=ABCMeta):

    @classmethod
    @lru_cache(maxsize=None)
    def get_regular_t(cls, n: int) -> torch.Tensor:
        """
        Subdivide the interval [0, 1] into `n` segments.

        Return:
            (`torch.Tensor`): Shape [n, 1].
        """
        return torch.linspace(0.0, 1.0, n).unsqueeze(-1)

    @abstractmethod
    def get_regular_vertex(self, num_segments: int) -> torch.Tensor:
        """
        Get `num_segments` evenly spaced points on the curve.

        Return:
            (`torch.Tensor`): Shape [m, d], where d is the dimension of the curve.
        """
        raise NotImplementedError

    @abstractmethod
    def get_regular_tangent(self, num_segments: int) -> torch.Tensor:
        """
        Get `num_segments` evenly spaced tangent vectors on the curve.

        Return:
            (`torch.Tensor`): Shape [m, d], where d is the dimension of the curve.
        """
        raise NotImplementedError

    @abstractmethod
    def get_regular_curvature(self, num_segments: int) -> torch.Tensor:
        """
        Get `num_segments` evenly spaced curvature vectors on the curve.

        Return:
            (`torch.Tensor`): Shape [m, d], where d is the dimension of the curve.
        """
        raise NotImplementedError
