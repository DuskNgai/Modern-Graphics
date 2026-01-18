from functools import lru_cache

import torch

from .surface import ParametricSurface


def spow(x, p):
    return torch.sign(x) * torch.abs(x) ** p

class SuperellipsoidSurface(ParametricSurface):

    def __init__(self, e1: float = 1.0, e2: float = 1.0) -> None:
        self.e1 = float(e1)
        self.e2 = float(e2)

    @classmethod
    @lru_cache
    def get_regular_uv(cls, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        return torch.cat(
            [
                torch.Tensor([[0.0, -0.5 * torch.pi]]),
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(0, 2 * torch.pi, num_u_segments + 1)[:-1],
                        torch.linspace(-0.5 * torch.pi, 0.5 * torch.pi, num_v_segments + 1)[1:-1],
                        indexing="xy",
                    ),
                    dim=-1,
                ).reshape(-1, 2),
                torch.Tensor([[0.0, 0.5 * torch.pi]]),
            ],
            dim=0,
        )

    @lru_cache
    def get_regular_vertex(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        u, v = self.get_regular_uv(num_u_segments, num_v_segments).unbind(-1)

        cos_u = torch.cos(u)
        sin_u = torch.sin(u)
        cos_v = torch.cos(v)
        sin_v = torch.sin(v)

        f = spow(cos_v, self.e1)
        g = spow(cos_u, self.e2)
        h = spow(sin_u, self.e2)
        k = spow(sin_v, self.e1)

        return torch.stack(
            [
                f * g,
                f * h,
                k,
            ],
            dim=-1,
        )

    @lru_cache
    def get_regular_face(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        faces = []

        n_u = num_u_segments
        n_v = num_v_segments

        # North pole
        for u in range(n_u):
            idx1 = u + 1
            idx2 = (u + 1) % n_u + 1
            faces.append([0, idx1, idx2])

        # Latitudes
        for v in range(n_v - 2):
            for u in range(n_u):
                idx0 = 1 + v * n_u + u
                idx1 = 1 + v * n_u + (u + 1) % n_u
                idx2 = 1 + (v + 1) * n_u + u
                idx3 = 1 + (v + 1) * n_u + (u + 1) % n_u

                faces.append([idx0, idx1, idx2])
                faces.append([idx1, idx3, idx2])

        # South pole
        for u in range(n_u):
            idx1 = 1 + (n_v - 2) * n_u + u
            idx2 = 1 + (n_v - 2) * n_u + (u + 1) % n_u
            faces.append([1 + (n_v - 1) * n_u, idx1, idx2])

        return torch.tensor(faces, dtype=torch.long)

    @lru_cache
    def get_regular_normal(self, num_u_segments: int, num_v_segments: int) -> torch.Tensor:
        uv = self.get_regular_uv(num_u_segments, num_v_segments)
        u, v = uv.unbind(-1)

        cos_u = torch.cos(u)
        sin_u = torch.sin(u)
        cos_v = torch.cos(v)
        sin_v = torch.sin(v)

        e1 = self.e1
        e2 = self.e2

        # signed power and absolute powers for derivatives
        def spow(x, p):
            return torch.sign(x) * torch.abs(x) ** p

        abs_cos_u = torch.abs(cos_u)
        abs_sin_u = torch.abs(sin_u)
        abs_cos_v = torch.abs(cos_v)
        abs_sin_v = torch.abs(sin_v)

        f = spow(cos_v, e1)
        g = spow(cos_u, e2)
        h = spow(sin_u, e2)
        k = spow(sin_v, e1)

        # derivatives wrt u
        gprime = -e2 * (abs_cos_u ** (e2 - 1)) * sin_u
        hprime = e2 * (abs_sin_u ** (e2 - 1)) * cos_u

        # derivatives wrt v
        fprime = -e1 * (abs_cos_v ** (e1 - 1)) * sin_v
        kprime = e1 * (abs_sin_v ** (e1 - 1)) * cos_v

        r_u = torch.stack([f * gprime, f * hprime, torch.zeros_like(u)], dim=-1)
        r_v = torch.stack([fprime * g, fprime * h, kprime], dim=-1)

        n = torch.cross(r_u, r_v, dim=-1)
        return n
