# Autonomous Driving W/ Deep Reinforcement Learning **(Ongoing Project)**

The objective is to develop human-like driving behaviors in autonomous vehicles using a hierarchical deep reinforcement learning (DRL) framework. Visual feature extraction and dimensionality reduction from RGB camera images are conducted using an autoencoder, which allows the agent to analyze the driving environment more effectively. Soft Actor-Critic (SAC) algorithm is selected as the DRL method for decision-making and control in autonomous driving; additionally, action noise is incorporated to facilitate a more efficient learning process. Action filtering techniques, such as Exponential Moving Average (EMA) and Weighted Moving Average (WMA), are employed to smooth the movements of the steering and throttle, thereby enhancing driving fluidity.

**The experiments carried out in this study aim to complete the course with a smooth human-like driving experience by making decisions and providing control with the proposed system, staying within the lane and at the targeted speed. The simulation environment required for the experiments was provided using the simulator of the Donkeycar project, an open-source autonomous vehicle driving platform.**

Some detailed information about the study is given below. The study is in the development process and will be edited in the near future.

Video to observe performances: [Video](https://www.youtube.com/watch?v=UJ_SdjIPPi8) 
____________________________________________
 | System Architecture For Training
 |:-------------------------:
 | ![Screenshot from 2024-10-07 12-44-58](https://github.com/user-attachments/assets/25889082-82de-4fc2-b3d4-198d7fdeee37)

Model Type | Description
:-------------------------:|:-------------------------:
SAC|Basic SAC (Reference)
NSAC| Action Noise Added SAC
NSAC-EMA| Noised + Action filtered SAC w/ Exponential moving average (EMA)
NSAC-WMA| Noised + Action filtered SAC w/ Weighted moving average (WMA)
