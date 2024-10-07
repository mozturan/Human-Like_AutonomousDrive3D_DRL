# Autonomous Driving W/ Deep Reinforcement Learning **(Ongoing Project)**

The objective is to develop human-like driving behaviors in autonomous vehicles using a hierarchical deep reinforcement learning (DRL) framework. Visual feature extraction and dimensionality reduction from RGB camera images are conducted using an autoencoder, which allows the agent to analyze the driving environment more effectively. Soft Actor-Critic (SAC) algorithm is selected as the DRL method for decision-making and control in autonomous driving; additionally, action noise is incorporated to facilitate a more efficient learning process. Action filtering techniques, such as Exponential Moving Average (EMA) and Weighted Moving Average (WMA), are employed to smooth the movements of the steering and throttle, thereby enhancing driving fluidity.

Autoencoder Results | System Architecture For Training
![sys](https://github.com/user-attachments/assets/65905a31-0289-42ca-a26d-a66947f0f13b)

Model Type | Description
:-------------------------:|:-------------------------:
SAC|Basic SAC (Reference)
NSAC| Action Noise Added SAV
NSAC-EMA| Noised + Action filtered SAC w/ Exponential moving average (EMA)
NSAC-WMA| Noised + Action filtered SAC w/ Weighted moving average (WMA)
