from functools import lru_cache

import torch

from .surface import ParametricSurface


class MobiusSurface(ParametricSurface):

    def __init__(self, radius: float = 1.0, width: float = 1.0) -> None:
        self.radius = radius
        self.width = width

    @classmethod
    @lru_cache
    def get_regular_uv(cls, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        return torch.stack(
            torch.meshgrid(
                torch.linspace(0, 2 * torch.pi, num_u_segments + 1)[:-1],
                torch.linspace(-0.5, 0.5, num_v_segments),
                indexing="xy",
            ),
            dim=-1,
        ).reshape(-1, 2)                                                  # [V * U, 2]

    @lru_cache
    def get_regular_vertex(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        u, v = self.get_regular_uv(num_u_segments, num_v_segments).unbind(-1)
        return torch.stack(
            [
                (self.radius + self.width * v / 2 * torch.cos(u / 2)) * torch.cos(u),
                (self.radius + self.width * v / 2 * torch.cos(u / 2)) * torch.sin(u),
                v / 2 * torch.sin(u / 2),
            ],
            dim=-1,
        )

    @lru_cache
    def get_regular_face(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        faces = []

        n_u = num_u_segments
        n_v = num_v_segments

        for v in range(n_v - 1):
            for u in range(n_u):
                idx0 = v * n_u + u
                idx2 = (v + 1) * n_u + u

                if u == n_u - 1:
                    idx1 = ((n_v - 1) - v) * n_u
                    idx3 = ((n_v - 1) - (v + 1)) * n_u
                else:
                    idx1 = v * n_u + (u + 1)
                    idx3 = (v + 1) * n_u + (u + 1)

                faces.append([idx0, idx1, idx2])
                faces.append([idx1, idx3, idx2])

        return torch.tensor(faces, dtype=torch.long)

    @lru_cache
    def get_regular_normal(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        u, v = self.get_regular_uv(num_u_segments, num_v_segments).unbind(-1)

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
