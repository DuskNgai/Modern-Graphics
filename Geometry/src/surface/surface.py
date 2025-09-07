from abc import (
    ABCMeta,
    abstractmethod,
)
from functools import lru_cache

import torch

__all__ = ['ParametricSurface']


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

    @lru_cache
    def get_regular_mesh(
        self,
        num_u_segments: int,
        num_v_segments: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_v_segments is None:
            vertex = self.get_regular_vertex(num_u_segments)
            face = self.get_regular_face(num_u_segments)
            normal = self.get_regular_normal(num_u_segments)
        else:
            vertex = self.get_regular_vertex(num_u_segments, num_v_segments)
            face = self.get_regular_face(num_u_segments, num_v_segments)
            normal = self.get_regular_normal(num_u_segments, num_v_segments)
        normal = normal / torch.norm(normal, dim=-1, keepdim=True)
        return vertex, face, normal
