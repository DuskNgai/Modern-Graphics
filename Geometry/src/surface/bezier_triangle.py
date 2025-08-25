from functools import lru_cache
import math

import torch

from .surface import ParametricSurface

__all__ = ['BezierTriangleSurface']


class BezierTriangleSurface(ParametricSurface):

    def __init__(self, degree: int, control_point: torch.Tensor) -> None:
        self.degree = degree
        self.control_point = torch.as_tensor(control_point) # [num_control_points, d]
        self.verify_shape_of_control_point()

        self.ijk = self.get_ijk(self.degree)                 # [num_control_points, 3]
        self.coefficient = self.get_coefficient(self.degree) # [n + 1, n + 1]

    def verify_shape_of_control_point(self) -> None:
        assert self.control_point.shape[-2] == (self.degree + 2) * (self.degree + 1) // 2

    @classmethod
    @lru_cache
    def get_ijk(cls, n: int) -> torch.LongTensor:
        """
        i + j + k = n, 0 <= i, j, k <= n

        Return:
            (`torch.Tensor`): Shape [num_control_points, 3].
        """
        ijk = []
        for i in range(n, -1, -1):
            for j in range(n - i, -1, -1):
                k = n - i - j
                ijk.append([i, j, k])
        return torch.tensor(ijk, dtype=torch.long)

    @classmethod
    @lru_cache
    def get_regular_uvw(cls, n: int) -> torch.Tensor:
        """
        u + v + w = 1, 0 <= u, v, w <= 1

        Return:
            (`torch.Tensor`): Shape [num_control_points, 3].
        """
        return cls.get_ijk(n) / n

    @classmethod
    @lru_cache
    def get_coefficient(cls, n: int) -> torch.Tensor:
        """
        n! / (i! * j! * k!)

        Return:
            (`torch.Tensor`): Shape [n + 1, n + 1].
        """
        coefficient = torch.zeros((n + 1, n + 1))
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                coefficient[i, j] = math.factorial(n) / (math.factorial(i) * math.factorial(j) * math.factorial(k))
        return coefficient

    @classmethod
    def get_bernstein(cls, coefficient: torch.Tensor, ijk: torch.LongTensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        n! / (i! * j! * k!) * (u ** i) * (v ** j) * (w ** k)

        Args:
            `coefficient` (`torch.Tensor`): Shape [n + 1, n + 1].
            `ijk` (`torch.LongTensor`): Shape [num_control_points, 3].
            `uvw` (`torch.Tensor`): Shape [..., 3].

        Return:
            (`torch.Tensor`): Shape [..., num_control_points]
        """
        i, j, k = ijk.unsqueeze(-3).unbind(-1) # [1, num_control_points]
        u, v, w = uvw.unsqueeze(-2).unbind(-1) # [..., 1]

        return coefficient[i, j] * (u ** i) * (v ** j) * (w ** k)

    @classmethod
    def get_d_bernstein_d_u(cls, coefficient: torch.Tensor, ijk: torch.LongTensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        n! / (i! * j! * k!) * [
            (i * u ** (i - 1)) * (v ** j) * (w ** k)
                -
            (u ** i) * (v ** j) * (k * w ** (k - 1))
        ]

        Args:
            `coefficient` (`torch.Tensor`): Shape [n + 1, n + 1].
            `ijk` (`torch.LongTensor`): Shape [num_control_points, 3].
            `uvw` (`torch.Tensor`): Shape [..., 3].

        Return:
            (`torch.Tensor`): Shape [..., num_control_points]
        """
        i, j, k = ijk.unsqueeze(-3).unbind(-1) # [1, num_control_points]
        u, v, w = uvw.unsqueeze(-2).unbind(-1) # [..., 1]

        term1 = torch.where(i > 0, i * (u ** (i - 1)), torch.zeros_like(u)) * (v ** j) * (w ** k)
        term2 = (u ** i) * (v ** j) * torch.where(k > 0, k * (w ** (k - 1)), torch.zeros_like(w))
        return coefficient[i, j] * (term1 - term2)

    @classmethod
    def get_d_bernstein_d_v(cls, coefficient: torch.Tensor, ijk: torch.LongTensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        n! / (i! * j! * k!) * [
            (u ** i) * (j * v ** (j - 1)) * (w ** k)
                -
            (u ** i) * (v ** j) * (k * w ** (k - 1))
        ]

        Args:
            `coefficient` (`torch.Tensor`): Shape [n + 1, n + 1].
            `ijk` (`torch.LongTensor`): Shape [num_control_points, 3].
            `uvw` (`torch.Tensor`): Shape [..., 3].

        Return:
            (`torch.Tensor`): Shape [..., num_control_points]
        """
        i, j, k = ijk.unsqueeze(-3).unbind(-1) # [1, num_control_points]
        u, v, w = uvw.unsqueeze(-2).unbind(-1) # [..., 1]

        term1 = (u ** i) * torch.where(j > 0, j * (v ** (j - 1)), torch.zeros_like(v)) * (w ** k)
        term2 = (u ** i) * (v ** j) * torch.where(k > 0, k * (w ** (k - 1)), torch.zeros_like(w))
        return coefficient[i, j] * (term1 - term2)

    @lru_cache
    def get_regular_vertex(self, num_segments_per_edge: int) -> torch.Tensor:
        """
        Return:
            (`torch.Tensor`): Shape [(m + 2) * (m + 1) // 2, d]
        """
        return self.evaluate(self.get_regular_uvw(num_segments_per_edge))

    @lru_cache
    def get_regular_face(self, num_segments_per_edge: int) -> torch.Tensor:
        """
        Return:
            (`torch.Tensor`): Shape [num_segments_per_edge * num_segments_per_edge, 3]
        """
        face = []
        for i in range(num_segments_per_edge):
            l, r = i * (i + 1) // 2, (i + 1) * (i + 2) // 2 # noqa: E741
            for j in range(l, r - 1):
                face.append([j, j + i + 1, j + i + 2])
                face.append([j + i + 2, j + 1, j])
            face.append([r - 1, r + i, r + i + 1])
        return torch.tensor(face, dtype=torch.long)         # [num_segments_per_edge * num_segments_per_edge, 3]

    @lru_cache
    def get_regular_normal(self, num_segments_per_edge: int) -> torch.Tensor:
        """
        Return:
            (`torch.Tensor`): Shape [(m + 2) * (m + 1) // 2, d]
        """
        return self.evaluate_normal(self.get_regular_uvw(num_segments_per_edge))

    def evaluate(self, uvw: torch.Tensor) -> torch.Tensor:
        """
        \sum_{i} \sum_{j} P[i, j, k] * [n! / (i! * j! * k!) * (u ** i) * (v ** j) * (w ** k)]

        Args:
            `uvw` (`torch.Tensor`): Shape [..., 3], u + v + w = 1

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        bernstein = self.get_bernstein(self.coefficient, self.ijk, uvw)          # [..., num_control_points]
        return torch.einsum("vc, ...cd -> ...vd", bernstein, self.control_point) # [..., d]

    def evaluate_normal(self, uvw: torch.Tensor) -> torch.Tensor:
        """
        \sum_{i} \sum_{j} P[i, j, k] * [n! / (i! * j! * k!) * [(i * u ** (i - 1)) * (v ** j) * (w ** k) - (u ** i) * (v ** j) * (k * w ** (k - 1))]]
            \\times
        \sum_{i} \sum_{j} P[i, j, k] * [n! / (i! * j! * k!) * [(u ** i) * (j * v ** (j - 1)) * (w ** k) - (u ** i) * (v ** j) * (k * w ** (k - 1))]]

        Args:
            `uvw` (`torch.Tensor`): Shape [..., 3], u + v + w = 1

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        d_bernstein_d_u = self.get_d_bernstein_d_u(self.coefficient, self.ijk, uvw)            # [..., num_control_points]
        d_bernstein_d_v = self.get_d_bernstein_d_v(self.coefficient, self.ijk, uvw)            # [..., num_control_points]
        d_vertex_d_u = torch.einsum("vc, ...cd -> ...vd", d_bernstein_d_u, self.control_point) # [..., d]
        d_vertex_d_v = torch.einsum("vc, ...cd -> ...vd", d_bernstein_d_v, self.control_point) # [..., d]
        return torch.cross(d_vertex_d_u, d_vertex_d_v, dim=-1)                                 # [..., d]
