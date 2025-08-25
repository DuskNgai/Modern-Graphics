import torch

from .curve import ParametricCurve

__all__ = ["BSplineCurve"]


class BSplineCurve(ParametricCurve):

    def __init__(self, degree: int, knot: torch.Tensor, control_point: torch.Tensor) -> None:
        self._degree = degree
        self._knot = knot
        self._control_point = control_point
