import numpy as np
import torch as th
from stable_baselines3 import PPO
from gym_donkeycar.envs.donkey_env import DonkeyEnv

# Load the saved model
model = PPO.load("ppo_donkey_model")

# Initialize the Donkey environment
env = DonkeyEnv("donkey-mountain-track-v0")  # Use the appropriate environment name

# Reset the environment
obs = env.reset()

# Run the agent in the simulator for a specified number of steps
for _ in range(1000):  # Adjust the number of steps as needed
    action, _ = model.predict(obs, deterministic=True)  # Use the model to predict actions
    obs, reward, done, info = env.step(action)

    # Optional: You can print or log the reward, done flag, or any other information
    print(f"Reward: {reward}, Done: {done}, Info: {info}")

    if done:
        obs = env.reset()  # Reset the environment if an episode is done

# Close the environment
env.close()
