import os
import numpy as np
import torch as th
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, ProgressBarCallback

from gym_donkeycar.envs.donkey_env import DonkeyEnv 

# Define a function to create the Donkey environment
def make_env():
    env = DonkeyEnv("donkey-mountain-track-v0")  # You can choose any Donkey environment here
    # env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_reward=10.0)
    return env

# Create the environment
env = make_env()

# Define the PPO model
model = PPO("MlpPolicy", env, verbose=1)


# Define the callback to evaluate the model and save checkpoints
eval_callback = EvalCallback(env, eval_freq=500, n_eval_episodes=5, deterministic=True, verbose=1)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./logs/", name_prefix="donkey_ppo")

# Train the model with callbacks
model.learn(total_timesteps=int(10000), progress_bar=True, callback=[eval_callback, checkpoint_callback])

# Save the trained model
model.save("ppo_donkey_model")

# Close the environment
env.close()
