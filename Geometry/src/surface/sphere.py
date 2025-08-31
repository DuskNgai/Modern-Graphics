from functools import lru_cache

import torch

from .surface import ParametricSurface


class SphereSurface(ParametricSurface):

    def __init__(self, radius: float = 1.0) -> None:
        self.radius = radius

    @classmethod
    @lru_cache
    def get_regular_uv(cls, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        u, v = torch.meshgrid(
            torch.linspace(0, 2 * torch.pi, n + 1),
            torch.linspace(0, torch.pi, n + 1),
            indexing="xy",
        )
        return u, v

    @lru_cache
    def get_regular_vertex(self, num_segments_per_edge: int) -> torch.Tensor:
        u, v = self.get_regular_uv(num_segments_per_edge)
        x = self.radius * torch.sin(v) * torch.cos(u)
        y = self.radius * torch.sin(v) * torch.sin(u)
        z = self.radius * torch.cos(v)
        return torch.stack([x, y, z], dim=-1).reshape(-1, 3)

    @lru_cache
    def get_regular_face(self, num_segments_per_edge: int) -> torch.Tensor:
        faces = []
        for i in range(num_segments_per_edge):
            for j in range(num_segments_per_edge):
                idx0 = i * (num_segments_per_edge + 1) + j
                idx1 = idx0 + 1
                idx2 = (i + 1) * (num_segments_per_edge + 1) + j
                idx3 = idx2 + 1
                faces.append([idx0, idx2, idx1])
                faces.append([idx1, idx2, idx3])
        return torch.tensor(faces, dtype=torch.long)

    @lru_cache
    def get_regular_normal(self, num_segments_per_edge: int) -> torch.Tensor:
        vertices = self.get_regular_vertex(num_segments_per_edge)
        return vertices / self.radius
