# Autonomous Driving W/ Deep Reinforcement Learning **(Ongoing Project)**

Autonomous driving task in a 3D simulation environment called "*DonkeyCar Sim*" using **Deep Reinforcement Learning** algortihms such as **Soft Actor-Critic (SAC)** and **Double Deep Q-Network with Prioritized Experience Replay (DDQN with PER)**.

## What is implemented?
- Both SAC and DDQN algortihms 
- Observation processing classes (kinematics and camera observation exist)
- Reward Shaping classes (different type of rewards exist)
- Custom Tesorboard
- CNN with AutoEncoders for pretraining the camera images
- CNN with VAE for pretraining the camera images (testing phase)
- 
## Notes, Issues and Todos
- In this phase of project agents are seems to able to learn but learning progress is very slow and poor, (See progress in examples)
- For Camera input, better feature extraction, an Autoencoder with CNNs is used.
- Stacking images for observation seems like not effecting learning progress.
- Reward functions probably not good enough. (New reward functions needed to be implemented)
- Hyperparameters optimization need (Optuna will be implemented)
- Autoencoder training with non-stacked images needed.
- Improvement needed for image preprocessing.

## Examples of learning progress

