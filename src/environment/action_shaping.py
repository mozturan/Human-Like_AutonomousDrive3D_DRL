import numpy as np

class SmoothingAction:
    def __init__(self, smoothing_coef = 0.3):
        
        self.smoothing_coef = smoothing_coef
        self.smoothed_action = None

    def reset(self):
        self.smoothed_action = None

    def step(self, action):
        if self.smoothed_action is None:
            self.smoothed_action = np.zeros_like(action)
        self.smoothed_action = self.smoothed_action * self.smoothing_coef + action * (1 - self.smoothing_coef)

        return self.smoothed_action

