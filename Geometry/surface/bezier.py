import math
import functools

import numpy as np

from surface import ParametricSurface

class BezierTriangle(ParametricSurface):
    def __init__(self,
        order: int,
        control_point: np.ndarray
    ) -> None:
        self.verify_shape_of_control_point(order, control_point)
        self.order = order
        self.control_point = control_point # [(n + 2) * (n + 1) // 2, d]

        self.ijk = self.get_ijk(self.order) # [(n + 2) * (n + 1) // 2, 3]
        self.coefficient = self.get_coefficient(self.order) # [n + 1, n + 1]

    def verify_shape_of_control_point(self, order: int, control_point: np.ndarray) -> None:
        assert control_point.shape[-2] == (order + 2) * (order + 1) // 2

    @classmethod
    @functools.cache(maxsize=16)
    def get_ijk(cls, n: int) -> np.ndarray:
        """
        i + j + k = n, 0 <= i, j, k <= n

        Return:
            (`np.ndarray`): Shape [(n + 2) * (n + 1) // 2, 3].
        """
        ijk = []
        for i in range(n, -1, -1):
            for j in range(n - i, -1, -1):
                ijk.append([i, j, n - i - j])
        return np.array(ijk)

    @classmethod
    def get_coefficient(cls, n: int) -> np.ndarray:
        """
        n! / (i! * j! * k!)

        Return:
            (`np.ndarray`): Shape [n + 1, n + 1].
        """
        coefficient = np.zeros((n + 1, n + 1), dtype=float)
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                coefficient[i, j] = math.factorial(n) / (math.factorial(i) * math.factorial(j) * math.factorial(k))
        return coefficient

    @classmethod
    def get_bernstein(cls, coefficient: np.ndarray, ijk: np.ndarray, uvw: np.ndarray) -> np.ndarray:
        """
        n! / (i! * j! * k!) * (u ** i) * (v ** j) * (w ** k)

        Args:
            coefficient (`np.ndarray`): Shape [n + 1, n + 1].
            ijk (`np.ndarray`): Shape [(n + 2) * (n + 1) // 2, 3].
            uvw (`np.ndarray`): Shape [..., 3].
        Return:
            (`np.ndarray`): Shape [..., (n + 2) * (n + 1) // 2]
        """
        i, j, k = np.split(ijk.transpose([-1, -2]), 3, axis=-2) # [1, (n + 2) * (n + 1) // 2]
        u, v, w = np.split(uvw, 3, axis=-1) # [..., 1]
        return coefficient[i, j] * (u ** i) * (v ** j) * (w ** k) # [..., (n + 2) * (n + 1) // 2]

    @classmethod
    def gererate_d_bernstein_d_u(cls, coefficient: np.ndarray, ijk: np.ndarray, uvw: np.ndarray) -> np.ndarray:
        """
        n! / (i! * j! * k!) * [(i * u ** (i - 1)) * (v ** j) * (w ** k) - (u ** i) * (v ** j) * (k * w ** (k - 1))]

        Return:
            (`np.ndarray`): Shape [..., (n + 2) * (n + 1) // 2]
        """
        i, j, k = np.split(ijk.transpose([-1, -2]), 3, axis=-2) # [1, (n + 2) * (n + 1) // 2]
        u, v, w = np.split(uvw, 3, axis=-1) # [..., 1]
        return coefficient[i, j] * (v ** j) * (
            np.nan_to_num(i * u ** (i - 1), nan=0) * (w ** k) - (u ** i) * np.nan_to_num(k * w ** (k - 1), nan=0)
        )

    @classmethod
    def gererate_d_bernstein_d_v(cls, coefficient: np.ndarray, ijk: np.ndarray, uvw: np.ndarray) -> np.ndarray:
        """
        n! / (i! * j! * k!) * [(u ** i) * (j * v ** (j - 1)) * (w ** k) - (u ** i) * (v ** j) * (k * w ** (k - 1))]

        Return:
            (`np.ndarray`): Shape [..., (n + 2) * (n + 1) // 2]
        """
        i, j, k = np.split(ijk.transpose([-1, -2]), 3, axis=-2) # [1, (n + 2) * (n + 1) // 2]
        u, v, w = np.split(uvw, 3, axis=-1) # [..., 1]
        return coefficient[i, j] * (u ** i) * (
            np.nan_to_num(j * v ** (j - 1), nan=0) * (w ** k) - (v ** j) * np.nan_to_num(k * w ** (k - 1), nan=0)
        )

    @classmethod
    @functools.cache(maxsize=16)
    def get_regular_uvw(cls, n: int) -> np.ndarray:
        """
        u + v + w = 1, 0 <= u, v, w <= 1

        Return:
            (`np.ndarray`): Shape [(n + 2) * (n + 1) // 2, 3].
        """
        return cls.get_ijk(n) / n

    def get_regular_vertex(self, num_segments_per_edge: int) -> np.ndarray:
        """
        Return:
            (`np.ndarray`): Shape [(m + 2) * (m + 1) // 2, d]
        """
        return self.evaluate(self.get_regular_uvw(num_segments_per_edge))

    def get_regular_face(self, num_segments_per_edge: int) -> np.ndarray:
        """
        Return:
            (`np.ndarray`): Shape [num_segments_per_edge * num_segments_per_edge, 3]
        """
        face = []
        for i in range(num_segments_per_edge):
            l, r = i * (i + 1) // 2, (i + 1) * (i + 2) // 2
            for j in range(l, r - 1):
                face.append([j, j + i + 1, j + i + 2])
                face.append([j, j + i + 2, j + 1])
            face.append([r - 1, r + i, r + i + 1])
        return np.array(face) # [num_segments_per_edge * num_segments_per_edge, 3]

    def get_regular_normal(self, num_segments_per_edge: int) -> np.ndarray:
        """
        Return:
            (`np.ndarray`): Shape [(m + 2) * (m + 1) // 2, d]
        """
        return self.evaluate_normal(self.get_regular_uvw(num_segments_per_edge))

    def evaluate(self, uvw: np.ndarray) -> np.ndarray:
        """
        \sum_{i} \sum_{j} P[i, j, k] * [n! / (i! * j! * k!) * (u ** i) * (v ** j) * (w ** k)]

        Args:
            uvw (`np.ndarray`): Shape [..., 3]

        Return:
            (`np.ndarray`): Shape [..., d]
        """
        bernstein = self.get_bernstein(self.coefficient, self.ijk, uvw) # [..., (n + 2) * (n + 1) // 2]
        return bernstein @ self.control_point # [..., d]

    def evaluate_normal(self, uvw: np.ndarray) -> np.ndarray:
        """
        \sum_{i} \sum_{j} P[i, j, k] * [n! / (i! * j! * k!) * [(i * u ** (i - 1)) * (v ** j) * (w ** k) - (u ** i) * (v ** j) * (k * w ** (k - 1))]]
            \\times
        \sum_{i} \sum_{j} P[i, j, k] * [n! / (i! * j! * k!) * [(u ** i) * (j * v ** (j - 1)) * (w ** k) - (u ** i) * (v ** j) * (k * w ** (k - 1))]]

        Args:
            uvw (`np.ndarray`): Shape [..., 3]

        Return:
            (`np.ndarray`): Shape [..., d]
        """
        d_bernstein_d_u = self.gererate_d_bernstein_d_u(self.coefficient, self.ijk, uvw) # [..., (n + 2) * (n + 1) // 2]
        d_bernstein_d_v = self.gererate_d_bernstein_d_v(self.coefficient, self.ijk, uvw) # [..., (n + 2) * (n + 1) // 2]
        d_vertex_d_u = d_bernstein_d_u @ self.control_point # [..., d]
        d_vertex_d_v = d_bernstein_d_v @ self.control_point # [..., d]
        return np.cross(d_vertex_d_u, d_vertex_d_v) # [..., d]
