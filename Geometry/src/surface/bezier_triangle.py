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

        self.ijk, self.ijk_to_index, self.index_to_ijk = self.get_ijk(self.degree) # [num_control_points, 3]
        self.coefficient = self.get_coefficient(self.degree)                       # [n + 1, n + 1]

    def verify_shape_of_control_point(self) -> None:
        assert self.control_point.shape[-2] == (self.degree + 2) * (self.degree + 1) // 2

    @classmethod
    @lru_cache
    def get_ijk(cls, n: int) -> torch.LongTensor:
        """
        i + j + k = n, 0 <= i, j, k <= n

        Return:
            (`torch.Tensor`): Shape [num_control_points, 3].
            (`dict`): Mapping from (i, j, k) to index.
            (`dict`): Mapping from index to (i, j, k).
        """
        ijk = []
        for i in range(n, -1, -1):
            for j in range(n - i, -1, -1):
                k = n - i - j
                ijk.append([i, j, k])
        ijk_to_index = {
            tuple(val): index
            for index, val in enumerate(ijk)
        }
        index_to_ijk = {
            index: tuple(val)
            for index, val in enumerate(ijk)
        }
        return torch.tensor(ijk, dtype=torch.long), ijk_to_index, index_to_ijk

    @classmethod
    @lru_cache
    def get_regular_uvw(cls, n: int) -> torch.Tensor:
        """
        u + v + w = 1, 0 <= u, v, w <= 1

        Return:
            (`torch.Tensor`): Shape [num_control_points, 3].
        """
        return cls.get_ijk(n)[0] / n

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

    def split(
        self,
        r: float,
        s: float,
        t: float,
    ) -> tuple["BezierTriangleSurface", "BezierTriangleSurface", "BezierTriangleSurface", "BezierTriangleSurface"]:
        """
        Splits the triangle into four smaller triangles: three corners and a central one.

        This method corresponds to splitting the three edges of the parametric domain.
        Let the original vertices be a (u-corner), b (v-corner), and c (w-corner).
        - The u-v edge (ab) is split by point f.
        - The v-w edge (bc) is split by point d.
        - The w-u edge (ca) is split by point e.
        
        The split parameters `r`, `s` and `t` define these points.
        - `f` is determined by `t`.
        - `d` is determined by `r`.
        - `e` is determined by `s`.

        This function returns the control nets for the four new triangles:
        1. u-corner: `afe`
        2. v-corner: `fbd`
        3. w-corner: `edc`
        4. central:  `def`

        Args:
            `r` (float): The split parameter for the v-w edge, in [0, 1].
            `s` (float): The split parameter for the w-u edge, in [0, 1].
            `t` (float): The split parameter for the u-v edge, in [0, 1].

        Returns:
            tuple[BezierTriangleSurface, ...]: A tuple containing the four new Bezier triangle surfaces,
                                               ordered as (u-corner, v-corner, w-corner, central).
        """
        assert 0 <= r <= 1, "Parameter 'r' must be in [0, 1]"
        assert 0 <= s <= 1, "Parameter 's' must be in [0, 1]"
        assert 0 <= t <= 1, "Parameter 't' must be in [0, 1]"

        vert_a = (1.0, 0.0, 0.0)
        vert_b = (0.0, 1.0, 0.0)
        vert_c = (0.0, 0.0, 1.0)

        vert_d = (0.0, 1.0 - r, r)
        vert_e = (s, 0.0, 1.0 - s)
        vert_f = (1.0 - t, t, 0.0)

        cp_u_corner = self._subdivide(vert_a, vert_f, vert_e)
        cp_v_corner = self._subdivide(vert_f, vert_b, vert_d)
        cp_w_corner = self._subdivide(vert_e, vert_d, vert_c)
        cp_centered = self._subdivide(vert_d, vert_e, vert_f)

        s_u = BezierTriangleSurface(self.degree, cp_u_corner)
        s_v = BezierTriangleSurface(self.degree, cp_v_corner)
        s_w = BezierTriangleSurface(self.degree, cp_w_corner)
        s_m = BezierTriangleSurface(self.degree, cp_centered)

        return s_u, s_v, s_w, s_m

    def _subdivide(self, v1: tuple[float, float, float], v2: tuple[float, float, float], v3: tuple[float, float, float]) -> torch.Tensor:

        @lru_cache
        def _get_de_casteljau_maps(n: int, device: torch.device) -> list[tuple[torch.Tensor, ...]]:
            maps = []
            for r in range(n):
                current_degree = n - r
                _, map_in, _ = self.get_ijk(current_degree)
                ijk_out, _, _ = self.get_ijk(current_degree - 1)

                indices_i, indices_j, indices_k = [], [], []
                for i, j, k in ijk_out.tolist():
                    indices_i.append(map_in[(i + 1, j, k)])
                    indices_j.append(map_in[(i, j + 1, k)])
                    indices_k.append(map_in[(i, j, k + 1)])

                maps.append((
                    torch.tensor(indices_i, dtype=torch.long, device=device),
                    torch.tensor(indices_j, dtype=torch.long, device=device),
                    torch.tensor(indices_k, dtype=torch.long, device=device),
                ))
            return maps

        n = self.degree
        device = self.control_point.device
        dtype = self.control_point.dtype

        maps = _get_de_casteljau_maps(n, device)

        # 2. 准备 "开花" 原理所需的 n 个参数
        # 我们要一次性计算所有新控制点，所以批次大小是 len(self.ijk)
        # args 的形状为 [batch_size, n, 3]
        # args[i, r, :] 是用于计算第 i 个新控制点时，在第 r 步所需的 (u,v,w)
        batch_size = self.ijk.shape[0]
        args = torch.zeros(batch_size, n, 3, device=device, dtype=dtype)
        v1_t, v2_t, v3_t = [torch.tensor(v, device=device, dtype=dtype) for v in (v1, v2, v3)]

        # 这个循环只用于构建参数矩阵，计算本身是并行的
        for i, (i_p, j_p, k_p) in enumerate(self.ijk):
            if i_p > 0:
                args[i, : i_p] = v1_t
            if j_p > 0:
                args[i, i_p : i_p + j_p] = v2_t
            if k_p > 0:
                args[i, i_p + j_p :] = v3_t

        # 3. 初始化控制点批次
        # temp_cp 的形状为 [batch_size, num_control_points, d]
        temp_cp = self.control_point.unsqueeze(0).expand(batch_size, -1, -1)

        # 4. 执行 n 步并行的德卡斯特里奥算法
        # 这个循环是串行的，但其内部所有操作都是在整个批次上并行执行的
        for r in range(n):
            # 获取当前步骤所需的 (u,v,w) 参数，形状 [batch_size, 3]
            uvw_params = args[:, r, :]
            u, v, w = uvw_params.unbind(-1) # u, v, w 的形状都是 [batch_size]

            # 获取当前步骤所需的索引映射
            indices_i, indices_j, indices_k = maps[r]

            # --- 这是向量化和并行化的核心 ---
            # 使用高级索引一次性收集所有需要计算的点
            # cp_i 的形状为 [batch_size, num_output_points, d]
            cp_i = temp_cp[:, indices_i, :]
            cp_j = temp_cp[:, indices_j, :]
            cp_k = temp_cp[:, indices_k, :]

            # 使用广播 (broadcasting) 进行并行计算
            # u,v,w 的形状从 [batch_size] 变为 [batch_size, 1, 1]
            # 以便能和 [batch_size, num_output_points, d] 形状的张量相乘
            temp_cp = (u.view(-1, 1, 1) * cp_i + v.view(-1, 1, 1) * cp_j + w.view(-1, 1, 1) * cp_k)
            # ------------------------------------

        # 5. 提取最终结果
        # 经过 n 步后, temp_cp 的形状为 [batch_size, 1, d]
        # 我们去掉中间维度为 1 的部分，得到 [batch_size, d]
        return temp_cp.squeeze(1)
