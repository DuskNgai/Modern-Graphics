from functools import lru_cache

import torch

from .surface import ParametricSurface


class MobiusSurface(ParametricSurface):

    def __init__(self, radius: float = 1.0, width: float = 1.0) -> None:
        self.radius = radius
        self.width = width

    @classmethod
    @lru_cache
    def get_regular_uv(cls, num_u_segments: int, num_v_segments: int) -> tuple[torch.Tensor, torch.Tensor]:
        u, v = torch.meshgrid(
            torch.linspace(0, 2 * torch.pi, num_u_segments + 1)[:-1],
            torch.linspace(-0.5, 0.5, num_v_segments + 1),
            indexing="ij",
        )
        return u, v

    @lru_cache
    def get_regular_vertex(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        u, v = self.get_regular_uv(num_u_segments, num_v_segments)
        v = v * self.width

        x = (self.radius + v / 2 * torch.cos(u / 2)) * torch.cos(u)
        y = (self.radius + v / 2 * torch.cos(u / 2)) * torch.sin(u)
        z = v / 2 * torch.sin(u / 2)

        return torch.stack([x, y, z], dim=-1).reshape(-1, 3)

    @lru_cache
    def get_regular_face(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        faces = []

        n_u = num_u_segments
        n_v = num_v_segments

        for i in range(n_u):
            for j in range(n_v):
                idx0 = i * (n_v + 1) + j
                idx2 = i * (n_v + 1) + (j + 1)

                if i == n_u - 1:
                    # Inverted triangle
                    idx1 = n_v - j
                    idx3 = n_v - (j + 1)
                else:
                    idx1 = (i + 1) * (n_v + 1) + j
                    idx3 = (i + 1) * (n_v + 1) + (j + 1)

                faces.append([idx0, idx1, idx2])
                faces.append([idx1, idx3, idx2])

        return torch.tensor(faces, dtype=torch.long)

    @lru_cache
    def get_regular_normal(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        u, v = self.get_regular_uv(num_u_segments, num_v_segments)

        dx_du = -torch.sin(u) * (self.radius + v / 2 * torch.cos(u / 2)) - v / 4 * torch.sin(u / 2) * torch.cos(u)
        dy_du = torch.cos(u) * (self.radius + v / 2 * torch.cos(u / 2)) - v / 4 * torch.sin(u / 2) * torch.sin(u)
        dz_du = v / 4 * torch.cos(u / 2)

        dx_dv = 1 / 2 * torch.cos(u / 2) * torch.cos(u)
        dy_dv = 1 / 2 * torch.cos(u / 2) * torch.sin(u)
        dz_dv = 1 / 2 * torch.sin(u / 2)

        nx = dy_du * dz_dv - dz_du * dy_dv
        ny = dz_du * dx_dv - dx_du * dz_dv
        nz = dx_du * dy_dv - dy_du * dx_dv

        return torch.stack([nx, ny, nz], dim=-1).reshape(-1, 3)
