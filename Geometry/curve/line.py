import torch

from .curve import ParametricCurve


class Line(ParametricCurve):

    def __init__(self, start_point: torch.Tensor, end_point: torch.Tensor) -> None:
        self.start_point = start_point
        self.end_point = end_point

    def get_regular_vertex(self, num_segments: int) -> torch.Tensor:
        """
        Return:
            (`torch.Tensor`): Shape [m, d]
        """
        t = self.get_regular_t(num_segments)
        return self.start_point + t * (self.end_point - self.start_point)

    def get_regular_tangent(self, num_segments: int) -> torch.Tensor:
        """
        Return:
            (`torch.Tensor`): Shape [m, d]
        """
        return (self.end_point - self.start_point).repeat(num_segments, 1)

    def get_regular_curvature(self, num_segments: int) -> torch.Tensor:
        """
        Return:
            (`torch.Tensor`): Shape [m, d]
        """
        return torch.zeros((num_segments, self.start_point.shape[-1]))
