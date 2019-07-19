import random

import gym
import numpy as np

import universe

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        # todo: duplication
        device = torch.device("cpu")

        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        # todo: duplication

        device = torch.device("cpu")

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(device)
        action = torch.tanh(mean + std * z)

        action = action.cpu()  # .detach().cpu().numpy()
        return action[0]


def update(replay_buffer, soft_q_net1, soft_q_net2, value_net, policy_net, target_value_net,
           batch_size, gamma=0.99, soft_tau=1e-2):
    # TODO: code duplication
    device = torch.device("cpu")

    value_criterion = nn.MSELoss()
    soft_q_criterion1 = nn.MSELoss()
    soft_q_criterion2 = nn.MSELoss()

    value_lr = 3e-4
    soft_q_lr = 3e-4
    policy_lr = 3e-4

    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
    soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
    soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)
    predicted_value = value_net(state)
    new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)

    # Training Q Function
    target_value = target_value_net(next_state)
    target_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())

    soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()
    # Training Value Function
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action), soft_q_net2(state, new_action))
    target_value_func = predicted_new_q_value - log_prob
    value_loss = value_criterion(predicted_value, target_value_func.detach())

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    # Training Policy Function
    policy_loss = (log_prob - predicted_new_q_value).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    display(anim)


def gif(policy_net):
    env = gym.make("Pendulum-v0")

    # Run a demo of the environment
    state = env.reset()
    cum_reward = 0
    frames = []
    for t in range(500):
        # Render into buffer.
        frames.append(env.render(mode='rgb_array'))
        action = policy_net.get_action(state)
        state, reward, done, info = env.step(action.detach())
        if done:
            break
    env.close()
    display_frames_as_gif(frames)


def sac_train():

    device = torch.device("cpu")
    env = NormalizedActions(gym.make("Pendulum-v0"))

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    hidden_dim = 256

    value_net = ValueNetwork(state_dim, hidden_dim).to(device)
    target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

    soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
    soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)

    replay_buffer_size = 1000000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # training

    max_frames = 100 #40000
    max_steps = 40 #500
    frame_idx = 0
    rewards = []
    batch_size = 10 #128

    while frame_idx < max_frames:
        #state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # todo: aggiunto io
            env.render()

            if frame_idx > 1000:
                action = policy_net.get_action(state).detach()
                next_state, reward, done, info = env.step(action.numpy())
            else:
                action = env.action_space.sample()  # take a random action
                next_state, reward, done, info = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            if len(replay_buffer) > batch_size:
                update(replay_buffer, soft_q_net1, soft_q_net2, value_net, policy_net, target_value_net,
                       batch_size)

            if frame_idx % 1000 == 0:
                plot(frame_idx, rewards)

            if done:
                break
        # todo: aggiunto io
        env.close()

        rewards.append(episode_reward)

    gif(policy_net)


def universe_train():
    env = gym.make('flashgames.NeonRace-v0')  # You can run many environment in parallel
    env.configure(remotes=1)  # automatically creates a local docker container
    observation_n = env.reset()  # Initiate the environment and get list of observations of its initial state
    while True:
        action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
        observation_n, reward_n, done_n, info = env.step(action_n)  # Reinforcement Learning action by agent
        print("observation: ", observation_n)  # Observation of the environment
        print("reward: ", reward_n)  # If the action had any postive impact +1/-1
        env.render()  # Run the agent on the environment


def stupid_train():
    env = gym.make('Pendulum-v0')
    for i_episode in range(10):
        observation = env.reset()
        for t in range(10):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


def main():

    # stupid_train()
    sac_train()
    # universe_train()


if __name__ == "__main__":
    main()
