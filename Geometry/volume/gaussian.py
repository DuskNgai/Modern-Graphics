import numpy as np

class Gaussian(object):
    def __init__(self,
        mean: np.ndarray,
        covariance: np.ndarray
    ) -> None:
        self.mean = mean
        self.covariance = covariance
