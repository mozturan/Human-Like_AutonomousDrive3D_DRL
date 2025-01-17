import argparse
import os
from typing import Tuple

import cv2
import gym
import gym_donkeycar  # noqa: F401
import numpy as np
import pygame
from pygame.locals import *  # noqa: F403

UP = (1, 0)
LEFT = (0, 1)
RIGHT = (0, -1)
DOWN = (-1, 0)

MAX_TURN = 1
MAX_THROTTLE = 0.5
# Smoothing constants
STEP_THROTTLE = 0.8
STEP_TURN = 0.8

frame_skip = 2
total_frames = 5000
render = True
output_folder = "./"

# Create folder if needed
os.makedirs(output_folder, exist_ok=True)

def control(
    x,
    theta: float,
    control_throttle: float,
    control_steering: float,
) -> Tuple[float, float]:
    """
    Smooth control.

    :param x:
    :param theta:
    :param control_throttle:
    :param control_steering:
    :return:
    """
    target_throttle = x * MAX_THROTTLE
    target_steering = MAX_TURN * theta
    if target_throttle > control_throttle:
        control_throttle = min(target_throttle, control_throttle + STEP_THROTTLE)
    elif target_throttle < control_throttle:
        control_throttle = max(target_throttle, control_throttle - STEP_THROTTLE)
    else:
        control_throttle = target_throttle

    if target_steering > control_steering:
        control_steering = min(target_steering, control_steering + STEP_TURN)
    elif target_steering < control_steering:
        control_steering = max(target_steering, control_steering - STEP_TURN)
    else:
        control_steering = target_steering
    return control_throttle, control_steering


# pytype: disable=name-error
moveBindingsGame = {K_UP: UP, K_LEFT: LEFT, K_RIGHT: RIGHT, K_DOWN: DOWN}  # noqa: F405
WHITE = (230, 230, 230)
pygame.font.init()
FONT = pygame.font.SysFont("Open Sans", 25)

pygame.init()
window = pygame.display.set_mode((400, 400), RESIZABLE)

control_throttle, control_steering = 0, 0
conf = {
    "port": 9091,
    "max_cte": 4.0,
    "cam_config":{"img_w": 160, "img_h": 120, "img_d": 3},
    "body_style": "car01",
    "body_rgb": [0,0,0],
    "car_name": "",
    "font_size": 20,
    "racer_name": "test",
    "country": "Middle East",
    "bio": "I am the best racer in the world",
    "host": "127.0.0.1",
    "lidar_config": {
        "deg_per_sweep_inc": 2.0,
        "deg_ang_down": 0.0,
        "deg_ang_delta": -1.0,
        "num_sweeps_levels": 1,
        "max_range": 20.0,
        "noise": 0.4,
        "offset_x": 0.0,
        "offset_y": 0.5,
        "offset_z": 0.5,
        "rot_x": 0.0
    }
}

env = gym.make("donkey-generated-track-v0", conf=conf)

lidar_data = []
t= 0
obs, reward, done, info = env.reset()
for frame_num in range(total_frames):
    x, theta = 0, 0
    # Record pressed keys
    keys = pygame.key.get_pressed()
    for keycode in moveBindingsGame.keys():
        if keys[keycode]:
            x_tmp, th_tmp = moveBindingsGame[keycode]
            x += x_tmp
            theta += th_tmp

    # Smooth control for teleoperation
    control_throttle, control_steering = control(x, theta, control_throttle, control_steering)

    window.fill((0, 0, 0))
    pygame.display.flip()
    # Limit FPS
    # pygame.time.Clock().tick(1 / TELEOP_RATE)
    for event in pygame.event.get():
        if (event.type == QUIT or event.type == KEYDOWN) and event.key in [  # pytype: disable=name-error
            K_ESCAPE,  # pytype: disable=name-error
            K_q,  # pytype: disable=name-error
        ]:
            env.close()
            exit()

    window.fill((0, 0, 0))
    text = "Control ready"
    text = FONT.render(text, True, WHITE)
    window.blit(text, (100, 100))
    pygame.display.flip()

    # steer, throttle
    action = np.array([-control_steering, control_throttle])

    for _ in range(frame_skip):
        obs, reward, done, info = env.step(action)
        if done:
            break
    if render:
        env.render()

    path = os.path.join(output_folder, f"{t}_{frame_num}.jpg")
    # Convert to BGR
    # cv2.imwrite(path, obs[:, :, ::-1])
    cv2.imwrite(path, obs)
    lidar_data.append(info["lidar"])

    if done:
        obs, reward, done, info = env.reset()
        control_throttle, control_steering = 0, 0
        #Save Lidar data as numpy

        t = t+1

np.save(os.path.join(output_folder, f"{t}_lidar_data.npy"), np.array(lidar_data))

env.close()




