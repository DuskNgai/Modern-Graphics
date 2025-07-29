import numpy as np

from curve import ParametricCurve

class Line(ParametricCurve):
    def __init__(self, start_point: np.ndarray, end_point: np.ndarray) -> None:
        self.start_point = start_point
        self.end_point = end_point

    def get_regular_vertex(self, num_segments: int) -> np.ndarray:
        """
        Return:
            (`np.ndarray`): Shape [m, d]
        """
        t = self.get_regular_t(num_segments)
        return self.start_point + t * (self.end_point - self.start_point)

    def get_regular_tangent(self, num_segments: int) -> np.ndarray:
        """
        Return:
            (`np.ndarray`): Shape [m, d]
        """
        return np.tile(self.end_point - self.start_point, (num_segments, 1))
    
    def get_regular_curvature(self, num_segments: int) -> np.ndarray:
        """
        Return:
            (`np.ndarray`): Shape [m, d]
        """
        return np.zeros((num_segments, self.start_point.shape[-1]))
