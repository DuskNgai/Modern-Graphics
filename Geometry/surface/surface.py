from abc import ABCMeta, abstractmethod

import numpy as np

class ParametricSurface(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def generate_regular_vertex(self, num_segments_per_edge: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def generate_regular_face(self, num_segments_per_edge: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def generate_regular_normal(self, num_segments_per_edge: int) -> np.ndarray:
        raise NotImplementedError

    def generate_regular_mesh(self, num_segments_per_edge: int) -> tuple[np.ndarray, np.ndarray]:
        vertex = self.generate_regular_vertex(num_segments_per_edge)
        face = self.generate_regular_face(num_segments_per_edge)
        normal = self.generate_regular_normal(num_segments_per_edge)
        normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
        uvw = self.generate_ijk(num_segments_per_edge) / num_segments_per_edge
        return vertex, face, normal, uvw
