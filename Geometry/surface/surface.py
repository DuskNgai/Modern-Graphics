from abc import ABCMeta, abstractmethod

import torch


class ParametricSurface(metaclass=ABCMeta):

    @abstractmethod
    def get_regular_vertex(self, num_segments_per_edge: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_regular_face(self, num_segments_per_edge: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_regular_normal(self, num_segments_per_edge: int) -> torch.Tensor:
        raise NotImplementedError

    def get_regular_mesh(self, num_segments_per_edge: int) -> tuple[torch.Tensor, torch.Tensor]:
        vertex = self.get_regular_vertex(num_segments_per_edge)
        face = self.get_regular_face(num_segments_per_edge)
        normal = self.get_regular_normal(num_segments_per_edge)
        normal = normal / torch.norm(normal, dim=-1, keepdims=True)
        uvw = self.get_ijk(num_segments_per_edge) / num_segments_per_edge
        return vertex, face, normal, uvw
