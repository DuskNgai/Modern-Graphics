from functools import lru_cache

import torch

from .curve import ParametricCurve

__all__ = ["BezierCurve"]


class BezierCurve(ParametricCurve):

    def __init__(self, control_point: torch.Tensor) -> None:
        """
        Args:
            `control_point` (`torch.Tensor`): Shape `[n + 1, d]`,
                where `n` is the degree of the curve,
                and `d` is the dimension of the curve.
        """
        self._control_point = torch.as_tensor(control_point, dtype=torch.float32) # [n + 1, d]
        self._degree = self._control_point.shape[0] - 1
        self._dimension = self._control_point.shape[1]

    @property
    @lru_cache(maxsize=1)
    def control_point(self) -> torch.Tensor:
        return self._control_point

    @property
    @lru_cache(maxsize=1)
    def degree(self) -> int:
        return self._degree

    @property
    @lru_cache(maxsize=1)
    def dimension(self) -> int:
        return self._dimension

    @lru_cache
    def get_regular_vertex(self, num_segments: int) -> torch.Tensor:
        return self.evaluate(self.get_regular_t(num_segments))

    @lru_cache
    def get_regular_tangent(self, num_segments: int) -> torch.Tensor:
        return self._evaluate_tangent(self.get_regular_t(num_segments))

    @lru_cache
    def get_regular_acceleration(self, num_segments: int) -> torch.Tensor:
        return self._evaluate_acceleration(self.get_regular_t(num_segments))

    @lru_cache
    def get_regular_curvature(self, num_segments: int) -> torch.Tensor:
        tangent = self.get_regular_tangent(num_segments)
        acceleration = self.get_regular_acceleration(num_segments)

        denominator = tangent.norm(dim=-1).pow(3) + 1e-8

        if self.dimension == 2:
            return torch.stack([tangent, acceleration], dim=-1).det().abs() / denominator
        elif self.dimension == 3:
            return torch.cross(tangent, acceleration, dim=-1).norm(dim=-1) / denominator
        else:
            raise ValueError("Curvature is only defined in 2D and 3D.")

    def evaluate(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bezier curve at parameter t.

        Args:
            `t` (`torch.Tensor`): Shape [...], values in [0, 1]

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        return self._evaluate_recurrent(t, 0, self.degree)

    def _evaluate_recurrent(self, t: torch.Tensor, s: int, e: int) -> torch.Tensor:
        """
        Recurrently evaluate Bezier curve at parameter t. O(degree) time complexity.
        """
        t = t.unsqueeze(-1)
        points = self._control_point[s : e + 1].unsqueeze(-2) # [e - s + 1, 1, d]
        for _ in range(e - s):
            points = torch.lerp(points[:-1], points[1 :], t)  # [e - s -> 1, ..., d]
        return points.squeeze(-3)                             # [..., d]

    def _evaluate_tangent(self, t: torch.Tensor) -> torch.Tensor:
        """
        C'(t) = n * {P[1 : n](t) - P[0 : n - 1](t)}

        Args:
            `t` (`torch.Tensor`): Shape [...], values in [0, 1]

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        return self.degree * (
            self._evaluate_recurrent(t, 1, self.degree) - \
            self._evaluate_recurrent(t, 0, self.degree - 1)
        )

    def _evaluate_acceleration(self, t: torch.Tensor) -> torch.Tensor:
        """
        C''(t) = n * (n - 1) * {P[2 : n](t) - 2 * P[1:n - 1](t) + P[0 : n - 2](t)}

        Args:
            `t` (`torch.Tensor`): Shape [...], values in [0, 1]

        Return:
            (`torch.Tensor`): Shape [..., d]
        """
        return self.degree * (self.degree - 1) * (
            self._evaluate_recurrent(t, 2, self.degree) - \
            self._evaluate_recurrent(t, 1, self.degree - 1) * 2 + \
            self._evaluate_recurrent(t, 0, self.degree - 2)
        )

    def split(self, u: float) -> tuple["BezierCurve", "BezierCurve"]:
        """
        Split Bezier curve at parameter `u`. O(degree) time complexity.

        Args:
            `u` (`float`): Value in [0, 1]

        Return:
            (`BezierCurve`, `BezierCurve`): Two Bezier curves.
        """
        assert 0.0 <= u <= 1.0, f"Splitting parameter u must be in [0, 1], but got {u}"

        u_tensor = torch.tensor(u, dtype=self.control_point.dtype, device=self.control_point.device)

        # return self._split_matrix(u_tensor)
        return self._split_de_casteljau(u_tensor)

    def _split_de_casteljau(self, u_tensor: torch.Tensor) -> tuple["BezierCurve", "BezierCurve"]:
        """
        Using De Casteljau's algorithm to split Bezier curve at parameter `u`.
        """
        cp = self.control_point
        l_cp, r_cp = [self.control_point[0]], [self.control_point[-1]]

        for _ in range(self.degree):
            cp = torch.lerp(cp[:-1], cp[1 :], u_tensor)
            l_cp.append(cp[0])
            r_cp.append(cp[-1])

        control_point_left = torch.stack(l_cp, dim=0)
        control_point_right = torch.stack(r_cp[::-1], dim=0)

        return BezierCurve(control_point_left), BezierCurve(control_point_right)

    def _split_matrix(self, u_tensor: torch.Tensor) -> tuple["BezierCurve", "BezierCurve"]:
        """
        A experimental implementation of splitting using matrix operations.
        Only triggered when `u = 0.5`.
        """

        @lru_cache
        def _get_binom_coeffs(n: int) -> torch.Tensor:
            """
            Computes a matrix of binomial coefficients, where binom[i, j] = C(i, j).
            """
            binom = torch.zeros((n + 1, n + 1), dtype=int)
            binom[0, 0] = 1
            for i in range(1, n + 1):
                binom[i, 0] = 1
                for j in range(1, i + 1):
                    binom[i, j] = binom[i - 1, j - 1] + binom[i - 1, j]
            return binom

        n = self.degree
        binom_coeffs = _get_binom_coeffs(n)

        i_indices = torch.arange(n + 1, device=self.control_point.device).view(-1, 1)
        j_indices = torch.arange(n + 1, device=self.control_point.device).view(1, -1)

        # m_l[i, j] = C[i, j] * u^j * v^{i - j}
        u_pow_l = u_tensor.pow(j_indices)
        v_pow_l = (1 - u_tensor).pow(i_indices - j_indices)
        m_l = torch.tril(binom_coeffs * u_pow_l * v_pow_l)

        rev_i = n - i_indices
        rev_j = n - j_indices

        # m_r[i, j] = C[n - i, j - i] * u^{j - i} * v^{n - j}, C[n - i, j - i] = C[n - i, n - j]
        u_pow_r = u_tensor.pow(j_indices - i_indices)
        v_pow_r = (1 - u_tensor).pow(rev_j)
        binom_r = binom_coeffs[rev_i, rev_j]
        m_r = torch.triu(binom_r * u_pow_r * v_pow_r)

        l_cp = m_l @ self.control_point
        r_cp = m_r @ self.control_point

        return BezierCurve(l_cp), BezierCurve(r_cp)

    def elevate(self) -> "BezierCurve":
        """
        Elevate one degree of the Bezier curve.

        Return:
            (`BezierCurve`): The elevated Bezier curve.
        """
        cp = self.control_point
        alpha = torch.arange(1, self.degree + 1, device=cp.device, dtype=cp.dtype).unsqueeze(1) / (self.degree + 1)
        cp = torch.cat([cp[: 1], torch.lerp(cp[1 :], cp[:-1], alpha), cp[-1 :]], dim=0)
        return BezierCurve(cp)
