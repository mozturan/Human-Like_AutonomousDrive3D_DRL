import numpy as np


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, 
                 dt=1e-2, x_initial=None, noise = True):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()
        self.noise = noise

    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)

    def __call__(self):

        if self.noise:
            x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
            )
            self.x_prev = x
        
        else:
            x = np.zeros_like(self.mean)
            
        return x

