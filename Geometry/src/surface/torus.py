from functools import lru_cache

import torch

from .surface import ParametricSurface


class TorusSurface(ParametricSurface):

    def __init__(self, major_radius: float = 1.0, minor_radius: float = 0.3) -> None:
        self.major_radius = major_radius
        self.minor_radius = minor_radius

    @classmethod
    @lru_cache
    def get_regular_uv(cls, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        u, v = torch.meshgrid(
            torch.linspace(0, 2 * torch.pi, n + 1),
            torch.linspace(0, 2 * torch.pi, n + 1),
            indexing="xy",
        )
        return u, v

    @lru_cache
    def get_regular_vertex(self, num_segments_per_edge: int) -> torch.Tensor:
        u, v = self.get_regular_uv(num_segments_per_edge)
        x = (self.major_radius + self.minor_radius * torch.cos(v)) * torch.cos(u)
        y = (self.major_radius + self.minor_radius * torch.cos(v)) * torch.sin(u)
        z = self.minor_radius * torch.sin(v)
        return torch.stack([x, y, z], dim=-1).reshape(-1, 3)

    @lru_cache
    def get_regular_face(self, num_segments_per_edge: int) -> torch.Tensor:
        faces = []
        n = num_segments_per_edge + 1

        for i in range(num_segments_per_edge):
            for j in range(num_segments_per_edge):
                idx0 = i * n + j
                idx1 = idx0 + 1
                idx2 = (i + 1) * n + j
                idx3 = idx2 + 1

                faces.append([idx0, idx2, idx1])
                faces.append([idx1, idx2, idx3])

        return torch.tensor(faces, dtype=torch.long)

    @lru_cache
    def get_regular_normal(self, num_segments_per_edge: int) -> torch.Tensor:
        u, v = self.get_regular_uv(num_segments_per_edge)
        nx = torch.cos(v) * torch.cos(u)
        ny = torch.cos(v) * torch.sin(u)
        nz = torch.sin(v)
        return torch.stack([nx, ny, nz], dim=-1).reshape(-1, 3)
