import gym
import numpy as np
import imageio

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC

env = gym.make('Pendulum-v0')
env = DummyVecEnv([lambda: env])

model = SAC(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000, log_interval=10)

model.save("../models/sac_pendulum")

del model  # remove to demonstrate saving and loading

model = SAC.load("../models/sac_pendulum")

#obs = env.reset()
#while True:
#    action, _states = model.predict(obs)
#    obs, rewards, dones, info = env.step(action)
#    env.render()
