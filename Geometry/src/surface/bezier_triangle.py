import functools
import math

import torch

from .surface import ParametricSurface

__all__ = ['BezierTriangleSurface']


class BezierTriangleSurface(ParametricSurface):

    def __init__(self, degree: int, control_point: torch.Tensor) -> None:
        self.verify_shape_of_control_point(degree, control_point)
        self.degree = degree
        self.control_point = control_point # [(n + 2) * (n + 1) // 2, d]

        self.ijk = self.get_ijk(self.degree)                 # [(n + 2) * (n + 1) // 2, 3]
        self.coefficient = self.get_coefficient(self.degree) # [n + 1, n + 1]

    def verify_shape_of_control_point(self, degree: int, control_point: torch.Tensor) -> None:
        assert control_point.shape[-2] == (degree + 2) * (degree + 1) // 2

    @classmethod
    @functools.cache
    def get_ijk(cls, n: int) -> torch.Tensor:
        """
        i + j + k = n, 0 <= i, j, k <= n

        Return:
            (`torch.Tensor`): Shape [(n + 2) * (n + 1) // 2, 3].
        """
        ijk = []
        for i in range(n, -1, -1):
            for j in range(n - i, -1, -1):
                ijk.append([i, j, n - i - j])
        return torch.tensor(ijk, dtype=torch.long)

    @classmethod
    def get_coefficient(cls, n: int) -> torch.Tensor:
        """
        n! / (i! * j! * k!)

        Return:
            (`torch.Tensor`): Shape [n + 1, n + 1].
        """
        coefficient = torch.zeros((n + 1, n + 1), dtype=torch.float64)
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                coefficient[i, j] = math.factorial(n) / (math.factorial(i) * math.factorial(j) * math.factorial(k))
        return coefficient

    @classmethod
    def get_bernstein(cls, coefficient: torch.Tensor, ijk: torch.Tensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        n! / (i! * j! * k!) * (u ** i) * (v ** j) * (w ** k)

        Args:
            coefficient (`torch.Tensor`): Shape [n + 1, n + 1].
            ijk (`torch.Tensor`): Shape [(n + 2) * (n + 1) // 2, 3].
            uvw (`torch.Tensor`): Shape [..., 3].
        Return:
            (`torch.Tensor`): Shape [..., (n + 2) * (n + 1) // 2]
        """
        i, j, k = ijk.t()                   # [(n + 2) * (n + 1) // 2]
        u, v, w = torch.unbind(uvw, dim=-1) # [...], [...], [...]
                                            # Expand for broadcasting
        i = i.unsqueeze(0)
        j = j.unsqueeze(0)
        k = k.unsqueeze(0)
        u = u.unsqueeze(-1)
        v = v.unsqueeze(-1)
        w = w.unsqueeze(-1)
        coeff = coefficient[i, j]
        return coeff * (u ** i) * (v ** j) * (w ** k)

    @classmethod
    def gererate_d_bernstein_d_u(cls, coefficient: torch.Tensor, ijk: torch.Tensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        n! / (i! * j! * k!) * [(i * u ** (i - 1)) * (v ** j) * (w ** k) - (u ** i) * (v ** j) * (k * w ** (k - 1))]

        Return:
            (`torch.Tensor`): Shape [..., (n + 2) * (n + 1) // 2]
        """
        i, j, k = ijk.t()
        u, v, w = torch.unbind(uvw, dim=-1)
        i = i.unsqueeze(0)
        j = j.unsqueeze(0)
        k = k.unsqueeze(0)
        u = u.unsqueeze(-1)
        v = v.unsqueeze(-1)
        w = w.unsqueeze(-1)
        coeff = coefficient[i, j]
        term1 = torch.where(i > 0, i * (u ** (i - 1)), torch.zeros_like(u)) * (v ** j) * (w ** k)
        term2 = (u ** i) * (v ** j) * torch.where(k > 0, k * (w ** (k - 1)), torch.zeros_like(w))
        return coeff * (term1 - term2)

    @classmethod
    def gererate_d_bernstein_d_v(cls, coefficient: torch.Tensor, ijk: torch.Tensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        n! / (i! * j! * k!) * [(u ** i) * (j * v ** (j - 1)) * (w ** k) - (u ** i) * (v ** j) * (k * w ** (k - 1))]

        Return:
            (`torch.Tensor`): Shape [..., (n + 2) * (n + 1) // 2]
        """
        i, j, k = ijk.t()
        u, v, w = torch.unbind(uvw, dim=-1)
        i = i.unsqueeze(0)
        j = j.unsqueeze(0)
        k = k.unsqueeze(0)
        u = u.unsqueeze(-1)
        v = v.unsqueeze(-1)
        w = w.unsqueeze(-1)
        coeff = coefficient[i, j]
        term1 = (u ** i) * torch.where(j > 0, j * (v ** (j - 1)), torch.zeros_like(v)) * (w ** k)
        term2 = (u ** i) * (v ** j) * torch.where(k > 0, k * (w ** (k - 1)), torch.zeros_like(w))
        return coeff * (term1 - term2)

    @classmethod
    @functools.cache
    def get_regular_uvw(cls, n: int) -> torch.Tensor:
        """
        u + v + w = 1, 0 <= u, v, w <= 1

        Return:
            (`torch.Tensor`): Shape [(n + 2) * (n + 1) // 2, 3].
        """
        return cls.get_ijk(n).to(torch.float64) / n

    def get_regular_vertex(self, num_segments_per_edge: int) -> torch.Tensor:
        """
        Return:
            (`torch.Tensor`): Shape [(m + 2) * (m + 1) // 2, d]
        """
        return self.evaluate(self.get_regular_uvw(num_segments_per_edge))

    def get_regular_face(self, num_segments_per_edge: int) -> torch.Tensor:
        """
        Return:
            (`torch.Tensor`): Shape [num_segments_per_edge * num_segments_per_edge, 3]
        """
        face = []
        for i in range(num_segments_per_edge):
            l, r = i * (i + 1) // 2, (i + 1) * (i + 2) // 2
            for j in range(l, r - 1):
                face.append([j, j + i + 1, j + i + 2])
                face.append([j, j + i + 2, j + 1])
            face.append([r - 1, r + i, r + i + 1])
        return torch.tensor(face, dtype=torch.long) # [num_segments_per_edge * num_segments_per_edge, 3]

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
            uvw (`torch.Tensor`): Shape [..., 3]

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        bernstein = self.get_bernstein(self.coefficient, self.ijk, uvw) # [..., (n + 2) * (n + 1) // 2]
        return torch.matmul(bernstein, self.control_point)              # [..., d]

    def evaluate_normal(self, uvw: torch.Tensor) -> torch.Tensor:
        """
        \sum_{i} \sum_{j} P[i, j, k] * [n! / (i! * j! * k!) * [(i * u ** (i - 1)) * (v ** j) * (w ** k) - (u ** i) * (v ** j) * (k * w ** (k - 1))]]
            \\times
        \sum_{i} \sum_{j} P[i, j, k] * [n! / (i! * j! * k!) * [(u ** i) * (j * v ** (j - 1)) * (w ** k) - (u ** i) * (v ** j) * (k * w ** (k - 1))]]

        Args:
            uvw (`torch.Tensor`): Shape [..., 3]

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        d_bernstein_d_u = self.gererate_d_bernstein_d_u(self.coefficient, self.ijk, uvw) # [..., (n + 2) * (n + 1) // 2]
        d_bernstein_d_v = self.gererate_d_bernstein_d_v(self.coefficient, self.ijk, uvw) # [..., (n + 2) * (n + 1) // 2]
        d_vertex_d_u = torch.matmul(d_bernstein_d_u, self.control_point)                 # [..., d]
        d_vertex_d_v = torch.matmul(d_bernstein_d_v, self.control_point)                 # [..., d]
        return torch.cross(d_vertex_d_u, d_vertex_d_v, dim=-1)                           # [..., d]
