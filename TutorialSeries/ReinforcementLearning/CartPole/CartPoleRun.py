import gym
import numpy
numpy.set_printoptions(threshold=numpy.nan)

env = gym.make("CartPole-v0")           #Reward when block broken
observation = env.reset()

print(observation)
print(observation.shape)
print(observation[0].shape)

for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # your agent here (this takes random actions)\
    print(action)
    observation, reward, done, info = env.step(action)
