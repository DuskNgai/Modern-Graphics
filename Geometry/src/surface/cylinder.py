from functools import lru_cache

import torch

from .surface import ParametricSurface


class CylinderSurface(ParametricSurface):

    def __init__(self, radius: float = 1.0, height: float = 2.0) -> None:
        self.radius = radius
        self.height = height

    @classmethod
    @lru_cache
    def get_regular_uv(cls, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        return torch.cat(
            [
                torch.Tensor([[0.0, 1.0]]),
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(0, 2 * torch.pi, num_u_segments + 1)[:-1],
                        torch.linspace(1, -1, num_v_segments + 1),
                        indexing="xy",
                    ),
                    dim=-1
                ).reshape(-1, 2),
                torch.Tensor([[0.0, -1.0]]),
            ],
            dim=0,
        )                                                                         # [V * U + 2, 2]

    @lru_cache
    def get_regular_vertex(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        u, v = self.get_regular_uv(num_u_segments, num_v_segments).unbind(-1)
        vertices = torch.stack(
            [
                self.radius * torch.cos(u),
                self.radius * torch.sin(u),
                self.height / 2 * v,
            ],
            dim=-1,
        )
        vertices[0, : 2] = 0.0
        vertices[-1, : 2] = 0.0

        return vertices

    @lru_cache
    def get_regular_face(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        faces = []
        n_u = num_u_segments
        n_v = num_v_segments

        # North Pole
        for u in range(n_u):
            idx1 = u + 1
            idx2 = (u + 1) % n_u + 1
            faces.append([0, idx2, idx1])

        # Latitudes
        for v in range(n_v):
            for u in range(n_u):
                idx0 = 1 + v * n_u + u
                idx1 = 1 + v * n_u + (u + 1) % n_u
                idx2 = 1 + (v + 1) * n_u + u
                idx3 = 1 + (v + 1) * n_u + (u + 1) % n_u

                faces.append([idx0, idx1, idx2])
                faces.append([idx1, idx3, idx2])

        # South Pole
        for u in range(n_u):
            idx1 = 1 + n_v * n_u + u
            idx2 = 1 + n_v * n_u + (u + 1) % n_u
            faces.append([1 + (n_v + 1) * n_u, idx1, idx2])

        return torch.tensor(faces, dtype=torch.long)

    @lru_cache
    def get_regular_normal(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        vertices = self.get_regular_vertex(num_u_segments, num_v_segments)

        ring_normals = vertices.clone()
        ring_normals[:, 2] = 0.0
        ring_normals = ring_normals / torch.norm(ring_normals, dim=-1, keepdim=True)

        normals = torch.zeros_like(vertices)
        normals[0] = torch.tensor([0.0, 0.0, 1.0])
        normals[1 :-1] = ring_normals[1 :-1]
        normals[-1] = torch.tensor([0.0, 0.0, -1.0])
        return normals
