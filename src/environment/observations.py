
# src/environment/observations.py
from abc import ABC, abstractmethod
import numpy as np
import cv2

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

    def __init__(self, stack_size=4, image_shape=None):
        # Implement for later
        # stack_size = 4
        # self.image_shape = image_shape
        # self.stack = np.zeros((stack_size, *image_shape, 1))
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
        self.stack_shape = (image_shape[0], image_shape[1], 
                            stack_size)
        print(self.stack.shape)

    def __call__(self, state):
        """
        Process the input and creates a new observation
        """
        self.stack = np.roll(self.stack, -1, axis=0)
        self.stack[-1] = self.rgb2gray(state)[:, :, np.newaxis]

        #perform histogram equalization
        self.stack[-1] = cv2.equalizeHist(self.stack[-1].astype(np.uint8))

        #normalize between 0 and 1
        self.stack[-1] = self.stack[-1]/255

    def reset(self, state):
        """
        Resets the stack with the first image only
        """
        self.stack = np.repeat(self.rgb2gray(state)[np.newaxis, :, :, np.newaxis], self.stack.shape[0], axis=0)
        
        #perform histogram equalization
        self.stack = cv2.equalizeHist(self.stack.astype(np.uint8))

        #normalize between 0 and 1
        self.stack = self.stack/255

    def reshape(self,input_shape, stack):
        stacked_images = stack.reshape((-1,) + input_shape)
        stacked_images = stacked_images.squeeze()
        return stacked_images
    
    def get_observation(self):
        return self.reshape(self.stack_shape, self.stack)
    

    # Returns a stack of grayscale images with shape (stack_size, image_shape[0], image_shape[1], 1) when called with a state

class ObservationStacker(Camera):
    def __init__(self, stack_size, image_shape=(120, 160)):
        self.stack_size = stack_size
        self.image_shape = image_shape
        self.stack = np.zeros((image_shape[0], image_shape[1], stack_size), dtype=np.uint8)
    def preprocess_image(self, image):
        # Ensure the image is of type uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Perform histogram equalization
        equalized_image = cv2.equalizeHist(gray_image)
        return equalized_image /255.0
    
    def add_observation(self, image):
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        # Add the new image to the stack, pop the oldest one
        self.stack = np.roll(self.stack, shift=-1, axis=2)
        self.stack[:, :, -1] = processed_image

    def reset(self, image):
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        # Fill the stack with the processed image
        self.stack = np.stack([processed_image] * self.stack_size, axis=2)

    def get_observation(self):
        return np.array(self.stack)

# def test_return_stack_with_state():
#         stack_size = 3
#         image_shape = (120, 160)
#         camera_stack = ObservationStacker(stack_size, image_shape)
#         state = np.ones((image_shape[0], image_shape[1], 3))
#         camera_stack.add_observation(state)
#         print(camera_stack.get_observation())
#         print("----------------------------")

#         camera_stack.add_observation(state)
#         print(camera_stack.get_observation())
#         print("----------------------------")

#         camera_stack.add_observation(state)
#         print(camera_stack.get_observation())
#         print("----------------------------")

#         camera_stack.add_observation(state)
#         print(camera_stack.get_observation())
#         print("----------------------------")

#         state = np.zeros((image_shape[0], image_shape[1], 3))
#         camera_stack.reset(state)
#         print(camera_stack.get_observation())
#         print("----------------------------")

#         state = np.ones((image_shape[0], image_shape[1], 3))
#         camera_stack.add_observation(state)
#         print(camera_stack.get_observation())

#         print(camera_stack.get_observation().shape)
#         # assert result.shape == (stack_size, image_shape[0], image_shape[1], 1)

#         import matplotlib.pyplot as plt

#         # state = np.ones((image_shape[0], image_shape[1], 3))
#         # camera_stack.reset(state)

#         # print(camera_stack.get_observation())
#         plt.imshow(camera_stack.get_observation()[:,:,2])
#         plt.show()
# test_return_stack_with_state()