import numpy as np
import os

class PerformanceMSE:
    def __init__(self, ref_cte=0.0, ref_speed = 2.0, name='mse'):
        self.name = name
        self.ref_cte = ref_cte
        self.ref_speed = ref_speed
        self.reset()

    def reset(self):
        self.total = 0
        self.count = 0
        self.mean = 0
        self.ctes = []
        self.speeds = []
        self.delta = 0
        self.last_action = None

    def __call__(self, cte, speed, action):

        self.ctes.append(abs(cte))
        self.speeds.append(speed)
        cte_error = (abs(cte) - self.ref_cte) ** 2
        speed_error = (speed - self.ref_speed) ** 2
        mean_error = (cte_error + speed_error) / 2.0

        self.total += mean_error

        if self.last_action is not None:
            self.delta += np.mean((np.array(action) - self.last_action) ** 2)

        self.last_action = np.array(action)
        self.count += 1

    def get_metrics(self):
        self.mean = self.total / self.count
        cte_avg = np.mean(self.ctes)
        # cte_std = np.std(self.ctes)
        # cte_median = np.median(self.ctes)
        # cte_max = np.max(self.ctes)

        speed_avg = np.mean(self.speeds)
        # speed_std = np.std(self.speeds)
        # speed_median = np.median(self.speeds)
        # speed_max = np.max(self.speeds)

        avg_delta = self.delta / self.count

        return self.mean, cte_avg, speed_avg, avg_delta