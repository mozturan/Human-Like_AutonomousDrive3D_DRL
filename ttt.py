import matplotlib.pyplot as plt
import math
import numpy as np
TARGET_SPEED = 1.
SIGMA =0.2
speed = 0

reward = math.exp((-(TARGET_SPEED- speed)**2)/(2*SIGMA**2))
#Create integer x_values from -10 to 10
x_values = np.arange(0, 5, 0.001) 
y_values = []
for i in range(len(x_values)):
    reward = math.exp(-(TARGET_SPEED - x_values[i])**2/(2*SIGMA**2))
    y_values.append(reward**0.5)

plt.plot(x_values,y_values)
plt.xlabel('Speed')
plt.ylabel('Reward')

plt.show()

import wandb

wandb.init(
    project = "tt",
    name = "ttttt")

# for i in range(3000):
#     wandb.log({"reward" : y_values[i]})


def _calculate_cte_reward(cte) -> float:
        
        if cte > 4.0 or cte < -4.0:
            return 0
        return (1.0 - (abs(cte)/4.0)**2)

cte_values = np.arange(-5.0, 5.0, 0.1) 
rewards = []
for i in range(len(cte_values)):
    reward = _calculate_cte_reward(cte_values[i])
    rewards.append(reward**0.5)


plt.plot(cte_values,rewards)
plt.show()

for i in range(len(cte_values)):
     wandb.log({"rr" : rewards[i]})