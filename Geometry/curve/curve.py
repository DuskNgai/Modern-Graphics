from abc import (
    ABCMeta,
    abstractmethod,
)
from functools import lru_cache

import numpy as np

class ParametricCurve(metaclass=ABCMeta):
    @classmethod
    @lru_cache(maxsize=None)
    def get_regular_t(cls, n: int) -> np.ndarray:
        """
        Subdivide the interval [0, 1] into `n` segments.

        Return:
            (`np.ndarray`): Shape [n, 1].
        """
        return np.linspace(0.0, 1.0, n)[..., np.newaxis]

    @abstractmethod
    def get_regular_vertex(self, num_segments: int) -> np.ndarray:
        """
        Get `num_segments` evenly spaced points on the curve.

        Return:
            (`np.ndarray`): Shape [m, d], where d is the dimension of the curve.
        """
        raise NotImplementedError

    @abstractmethod
    def get_regular_tangent(self, num_segments: int) -> np.ndarray:
        """
        Get `num_segments` evenly spaced tangent vectors on the curve.

        Return:
            (`np.ndarray`): Shape [m, d], where d is the dimension of the curve.
        """
        raise NotImplementedError

    @abstractmethod
    def get_regular_curvature(self, num_segments: int) -> np.ndarray:
        """
        Get `num_segments` evenly spaced curvature vectors on the curve.

        Return:
            (`np.ndarray`): Shape [m, d], where d is the dimension of the curve.
        """
        raise NotImplementedError
