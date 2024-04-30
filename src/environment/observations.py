
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


class Kinematics(ObservationType):
    """
    Class for creating a kinematic observation
    """


    def __init__(self, info_config=None):
        """
        Constructor
        info_config: list of str, values from info dict which should be used in the observation
        If None, uses the default values: ['pos', 'cte', 'speed', 'gyro', 'accel', 'vel']

        * Default values:
        INFO:
            {'pos': (x,y,z), 
            'cte': float, 
            'speed': float, 
            'forward_vel': float, 
            'hit': 'none', 
            'gyro': (x,y,z), 
            'accel': (x,y,z), 
            'vel': (x,y,z), 
        
        * Optional values:
        INFO:
            {'pos': (x,y,z), 
            'cte': float, 
            'speed': float, 
            'forward_vel': float, 
            'hit': 'none', 
            'gyro': (x,y,z), 
            'accel': (x,y,z), 
            'vel': (x,y,z), 
            'lidar': [], 
            'car': (x,y,z),
            'last_lap_time': 0.0, 
            'lap_count': 0}        
            }
        """

        self.info_config = \
            info_config if info_config else ['pos', 
                                             'cte', 
                                             'speed', 
                                             'gyro', 
                                             'accel', 
                                             'vel']
    
    def __call__(self, action, info) -> np.ndarray:
        """
        Process the input and creates a new observation
        """
        
        values = []
        for key in self.info_config:
            value = info[key]
            if isinstance(value, tuple):
                values.extend(list(value))
            else:
                values.append(value)

        for act in action:
            values.append(act)

        return np.array(values)


class Camera(ObservationType):
    """
    Class for creating a camera observation
    """

    def __init__(self, stack_size, image_shape=None):
        # Implement for later
        stack_size = 4
        self.image_shape = image_shape
        self.stack = np.zeros()
        pass

    def __call__(self, state):
        """
        Process the input and creates a new observation
        """
        image = self.rgb2gray(state)
        image = np.expand_dims(image, axis=-1)
        return image

    def rgb2gray(self, rgb):
        """
        Converts an RGB image to grayscale
        """
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


class CameraStack(Camera):
    """
    Class for stacking sequential grayscale images
    """

    def __init__(self, stack_size = 4, image_shape = (120,160)):
        super().__init__(stack_size, image_shape)
        self.stack = np.zeros((stack_size, *image_shape, 1))

    def __call__(self, state):
        """
        Process the input and creates a new observation
        """
        self.stack = np.roll(self.stack, -1, axis=0)
        self.stack[-1] = self.rgb2gray(state)[:, :, np.newaxis]
        return self.stack

    def reset(self, state):
        """
        Resets the stack with the first image only
        """
        self.stack[0] = self.rgb2gray(state)[:, :, np.newaxis]
        self.stack[1:] = 0.

