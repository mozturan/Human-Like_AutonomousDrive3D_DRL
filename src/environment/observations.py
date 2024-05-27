# src/environment/observations.py
from abc import ABC, abstractmethod
import numpy as np
from scipy.ndimage import gaussian_filter
from keras.models import model_from_json as load
import os
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

    def __init__(self, num_history=2):
        # LOAD THE AE MODEL
        self.encoder = self.load_ae()
        self.encoder.summary()
        self.num_history = num_history
        self.observation_shape = (16,)
        self.last_observation = None
        self.last_info = None
        pass

    def reset(self, observation):
        self.last_observation = observation
        self.last_observation = self.crop_image(self.last_observation)
        self.last_observation = self.preprocess_image(self.last_observation)
        self.last_observation = self.reduction(self.last_observation)
        self.last_observation = self.last_observation[0]
        self.last_info = np.array([0.0,0.0,0.0,0.0])

        return self.history(self.last_observation, self.last_info)

    def history(self, encoded, info):
        return np.concatenate((encoded, self.last_observation, 
                               info, self.last_info), axis=0)
    
    def __call__(self, image, action, info):
        """
        Process the input and creates a new observation
        """
        info = self.info_edit(info, action)
        image = self.crop_image(image)
        image = self.preprocess_image(image)
        encoded = self.reduction(image)
        history = self.history(encoded[0], info)

        self.last_info = info
        self.last_observation = encoded[0]

        return history

    def crop_image(self, image):
        return image[40:120, :, :]
        
    def preprocess_image(self, image):
        """
        Preprocess the image by normalizing it and converting it to uint8
        """
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        return image / 255.0

    def load_ae(self, model_folder="models/encoder/"):
        """
        Loads the encoder model from a json file and the weights from a h5 file
        """

        encoder_file = os.path.join(model_folder) + "encoder_model.json"
        weights_file = os.path.join(model_folder) + "encoder_weights.h5"

        with open(encoder_file, "r") as json_file:
            encoder_json = json_file.read()
        encoder = load(encoder_json)
        encoder.load_weights(weights_file)

        return encoder

    def reduction(self, image):
        image= np.expand_dims(image, axis=0)
        image = self.encoder.predict(image, verbose=0)
        return image
    
    def info_edit(self, info, action):
        return np.array([info["speed"],info["forward_vel"],action[0], action[1]])