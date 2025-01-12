import rooms
import gymnasium as gym
import matplotlib.pyplot as plot
from agent import RandomAgent, PolAgent
import numpy as np

# Setup with:
#> pip install gymnasium matplotlib torch pygame
# The /rooms folder must be in the same directory.

def run_episode(environment, agent, nr_episode, all_rewards, render=False, train=False):
    state, _ = environment.reset()
    if render:
        environment.render()
    episode_reward = agent.run(state, train, render, all_rewards)
    if nr_episode % 100 == 0:
        print(nr_episode, ":", episode_reward)
    return episode_reward

env = gym.make("rooms9-fixed")

params = dict()
params["nr_actions"] = env.action_space.n
params["obs_space"] = 4
params["alpha"] = 0.01  # learning rate
params["gamma"] = 0.99  # discount factor
params["epsilon"] = 0.75  # exploration rate (at the start of learning)
params["epsilon_min"] = 0.01  # minimal exploration rate
params["epsilon_decay"] = 0.0001  # epsilon decay
params["training_episodes"] = 2500  # training duration
params["demonstration_episodes"] = 2  # demonstration duration
params["env"] = env
params["render"] = True

ag = RandomAgent(params) # (random performance for comparison)
#ag = PolAgent(params) # TODO: 0

all_rewards = []

# train the agent
training_results = [run_episode(env, ag, ep, all_rewards, render=False, train=True) for ep in range(params["training_episodes"])]

# Plot episode reward
x = range(params["training_episodes"])
y = training_results
plot.plot(x, y)
plot.title("training progress")
plot.xlabel("episode")
plot.ylabel("episode reward")
plot.show()


# Mean over every 10 episodes
w_size = 10
x = range(0, len(training_results), w_size)
y = [np.mean(training_results[i:i + w_size]) for i in range(0, len(training_results), w_size)]
plot.plot(x, y)
plot.title('training progress')
plot.xlabel('episode')
plot.ylabel('mean reward (Over 10 Episodes)')
plot.show()

# let the agent demonstrate the learned policy
for ep in range(params["demonstration_episodes"]):
    run_episode(env, ag, ep, all_rewards, render=True, train=False)

