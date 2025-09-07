from functools import lru_cache

import torch

from .surface import ParametricSurface


class SphereSurface(ParametricSurface):

    def __init__(self, radius: float = 1.0) -> None:
        self.radius = radius

    @classmethod
    @lru_cache
    def get_regular_uv(cls, num_u_segments: int, num_v_segments: int) -> tuple[torch.Tensor, torch.Tensor]:
        u, v = torch.meshgrid(
            torch.linspace(0, 2 * torch.pi, num_u_segments + 1)[:-1],
            torch.linspace(0, torch.pi, num_v_segments + 1),
            indexing="ij",
        )
        return u, v

    @lru_cache
    def get_regular_vertex(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        u, v = self.get_regular_uv(num_u_segments, num_v_segments)
        x = self.radius * torch.sin(v) * torch.cos(u)
        y = self.radius * torch.sin(v) * torch.sin(u)
        z = self.radius * torch.cos(v)
        return torch.stack([x, y, z], dim=-1).reshape(-1, 3)

    @lru_cache
    def get_regular_face(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        faces = []

        n_u = num_u_segments
        n_v = num_v_segments

        for i in range(n_u):
            for j in range(n_v):
                idx0 = i * (n_v + 1) + j
                idx1 = ((i + 1) % n_u) * (n_v + 1) + j
                idx2 = i * (n_v + 1) + (j + 1)
                idx3 = ((i + 1) % n_u) * (n_v + 1) + (j + 1)

                if j == 0:
                    # North pole cap (triangles)
                    # The vertices with j = 0 are all at the same physical point.
                    # We form a triangle with two vertices from the next latitude ring.
                    faces.append([idx0, idx3, idx2])
                elif j == n_v - 1:
                    # South pole cap (triangles)
                    # The vertices with j = n_v - 1 are all at the same physical point.
                    # We form a triangle with two vertices from the previous latitude ring.
                    faces.append([idx0, idx1, idx3])
                else:
                    # Middle section (quads made of two triangles)
                    faces.append([idx0, idx1, idx2])
                    faces.append([idx1, idx3, idx2])

        return torch.tensor(faces, dtype=torch.long)

    @lru_cache
    def get_regular_normal(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        vertices = self.get_regular_vertex(num_u_segments, num_v_segments)
        return vertices / self.radius
