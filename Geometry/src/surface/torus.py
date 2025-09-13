from functools import lru_cache

import torch

from .surface import ParametricSurface


class TorusSurface(ParametricSurface):
    """
    Torus is regarded as two ellipsoids, one major ellipsoid and one minor ellipsoid.
    """

    def __init__(
        self,
        major_radius_x: float = 1.5,
        major_radius_y: float = 1.0,
        minor_radius_x: float = 0.4,
        minor_radius_y: float = 0.2,
    ) -> None:
        self.major_radius_x = major_radius_x
        self.major_radius_y = major_radius_y
        self.minor_radius_x = minor_radius_x
        self.minor_radius_y = minor_radius_y

    @classmethod
    @lru_cache
    def get_regular_uv(cls, num_u_segments: int, num_v_segments: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.stack(
            torch.meshgrid(
                torch.linspace(0, 2 * torch.pi, num_u_segments + 1)[:-1],
                torch.linspace(0, 2 * torch.pi, num_v_segments + 1)[:-1],
                indexing="xy",
            ),
            dim=-1,
        ).reshape(-1, 2)                                                  # [V * U, 2]

    @lru_cache
    def get_regular_vertex(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        u, v = self.get_regular_uv(num_u_segments, num_v_segments).unbind(-1)
        return torch.stack(
            [
                (self.major_radius_x + self.minor_radius_x * torch.cos(v)) * torch.cos(u),
                (self.major_radius_y + self.minor_radius_x * torch.cos(v)) * torch.sin(u),
                self.minor_radius_y * torch.sin(v),
            ],
            dim=-1,
        )

    @lru_cache
    def get_regular_face(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        faces = []

        n_u = num_u_segments
        n_v = num_v_segments

        for v in range(n_v):
            for u in range(n_u):
                idx0 = v * n_u + u
                idx1 = v * n_u + (u + 1) % n_u
                idx2 = ((v + 1) % n_v) * n_u + u
                idx3 = ((v + 1) % n_v) * n_u + (u + 1) % n_u

                faces.append([idx0, idx1, idx2])
                faces.append([idx1, idx3, idx2])

        return torch.tensor(faces, dtype=torch.long)

    @lru_cache
    def get_regular_normal(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        u, v = self.get_regular_uv(num_u_segments, num_v_segments).unbind(-1)
        a, b = self.major_radius_x, self.major_radius_y
        c, d = self.minor_radius_x, self.minor_radius_y

        cos_u, sin_u = torch.cos(u), torch.sin(u)
        cos_v, sin_v = torch.cos(v), torch.sin(v)

        nx = d * (b + c * cos_v) * cos_u * cos_v
        ny = d * (a + c * cos_v) * sin_u * cos_v
        nz = c * sin_v * ((a + c * cos_v) * sin_u ** 2 + (b + c * cos_v) * cos_u ** 2)
        return torch.stack([nx, ny, nz], dim=-1).reshape(-1, 3)
