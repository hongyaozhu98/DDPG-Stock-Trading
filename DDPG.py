import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines.common.vec_env import SubprocVecEnv

env = gym.make('stocks-v0')

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

# Hyper parameters

GAMMA = 0.99
TAU = 0.001
BATCH_SIZE = 250
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
BUFFER_SIZE = 50000

model = DDPG(MlpPolicy, env, 
              gamma = GAMMA, tau = TAU, batch_size = BATCH_SIZE,
              actor_lr = ACTOR_LEARNING_RATE, critic_lr = CRITIC_LEARNING_RATE,
              buffer_size = BUFFER_SIZE, verbose=1, 
              param_noise=param_noise, action_noise=action_noise)
model.learn(total_timesteps=25000)
model.save("ddpg_stock")

del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_stock")

test = gym.make('stockstest-v0')
obs = test.reset()

# Two years of testing

for i in range(505):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = test.step(action)
    test.render()
