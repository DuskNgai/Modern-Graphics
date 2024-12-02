from abc import ABCMeta, abstractmethod

import numpy as np

class ParametricCurve(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def generate_regular_vertex(self, num_segments: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def generate_regular_tangent(self, num_segments: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def generate_regular_curvature(self, num_segments: int) -> np.ndarray:
        raise NotImplementedError
