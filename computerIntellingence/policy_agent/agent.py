import random
import numpy
import numpy as np
import torch as th
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class Agent:
    """ Base class of an autonomously acting and learning agent. """

    def __init__(self, params):
        self.nr_actions = params["nr_actions"]

    def policy(self, state):
        """ Behavioral strategy of the agent. Maps states to actions. """
        pass

    def update(self, state, action, reward, next_state, done):
        """ Learning method of the agent. Integrates experience into the agent's current knowledge. """
        pass


class RandomAgent(Agent):
    """ Randomly acting agent. """

    def __init__(self, params):
        super(RandomAgent, self).__init__(params)
        self.env = params["env"]
        self.gamma = params["gamma"]

    def policy(self, state):
        return random.choice(range(self.nr_actions))

    def run(self, state, train, render, _):
        episode_reward = 0
        time_step = 0
        done = False
        while not done:
            # 1. Select action according to policy
            action = self.policy(state)
            # 2. Execute selected action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward * (self.gamma ** time_step)
            time_step += 1
            if render:
                self.env.render()
        return episode_reward

class PolAgent(Agent):
    """  Autonomous agent using Reinforce. """

    def __init__(self, params):
        super(PolAgent, self).__init__(params)
        self.all_rewards = []
        self.all_disc_ret = []
        self.gamma = params["gamma"]
        self.obs_dims = params["obs_space"]
        self.act_dims = params["nr_actions"]
        self.gamma = params["gamma"]
        self.policy_network = self.PolicyNetwork(self.obs_dims, 50, self.act_dims)
        self.optimizer = optim.Adam  # TODO: 1 initialize optimizer
        self.env = params["env"]

    def policy(self, state):
        state_tensor = th.tensor(state)
        act, action_probabilities = None  # TODO: 2 compute forward pass
        return act, action_probabilities

    def update(self, state, action, rewards, log_probs):
        disc_rew = self.discounted_reward(rewards)
        loss = self.policy_network.loss(log_probs,disc_rew)

        # TODO: 4 update network parameters (3 steps)


    def discounted_reward(self, rewards, discount_factor=0.99):
        #Source: https://stackoverflow.com/questions/65233426/discount-reward-in-reinforce-deep-reinforcement-learning-algorithm
        t_steps = np.arange(len(rewards))
        r = rewards * discount_factor ** t_steps
        r = r[::-1].cumsum()[::-1] / discount_factor ** t_steps
        return r

    def run(self, state, _, render, all_rewards):
        rewards = []
        log_probs = []
        discounted_return = 0
        done = False
        time_step = 0
        while not done:
            # 1. Select action according to policy
            action, action_prob = self.policy(state)
            log_probs.append(action_prob)
            # 2. Execute selected action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            discounted_return += reward * (self.gamma ** time_step)
            rewards.append(reward)
            if done:
                # 3. Integrate new experience into agent
                self.update(state, action, rewards, log_probs)
                all_rewards.append(np.sum(rewards))
                break
            state = next_state
            time_step += 1
            if render:
                self.env.render()

        return discounted_return

    class PolicyNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            action_probabilities = F.softmax(x, dim=0)
            action_distribution = th.distributions.Categorical(action_probabilities)
            action = action_distribution.sample()
            log_proba = action_distribution.log_prob(action)
            return action, log_proba

        def loss(self, log_probs, disc_rew):
            log_probs = th.stack(log_probs)
            discounted_rewards = th.tensor(disc_rew)
            loss = None  # TODO: 3 compute policy loss
            return loss





