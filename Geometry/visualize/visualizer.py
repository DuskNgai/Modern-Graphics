from abc import (
    ABCMeta,
    abstractmethod,
)

__all__ = ["Visualizer"]


class Visualizer(metaclass=ABCMeta):

    @abstractmethod
    def visualize(self):
        raise NotImplementedError
