import gym
import numpy as np
#numpy.set_printoptions(threshold=numpy.nan)

env = gym.make("Breakout-v0")           #Reward when block broken
observation = env.reset()

print(observation)
print(observation.shape)
print(observation[0].shape)
print(np.ravel(observation).shape)

for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # your agent here (this takes random actions)\
    observation, reward, done, info = env.step(action)
