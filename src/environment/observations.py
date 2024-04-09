
# src/environment/observations.py
from abc import ABC, abstractmethod
import numpy as np

class ObservationType(ABC):

    @abstractmethod
    def __call__(self, state, action, info):
        """
        Process the input and creates a new observation
        """
        pass



class KinematicsObservationType(ObservationType):
    """
    Class for creating a kinematic observation
    """

    def __call__(self, state, action, info):
        """
        Process the input and creates a new observation
        """
        return np.array([state.x, state.y, state.yaw, state.speed])



class Camera(ObservationType):
    """
    Class for creating a camera observation
    """

    def __init__(self, stack_size=4):
        self.stack_size = stack_size

    def __call__(self, state, action, info):
        """
        Process the input and creates a new observation
        """
        stacked_image = np.stack([state.image] * self.stack_size, axis=2)
        return stacked_image


