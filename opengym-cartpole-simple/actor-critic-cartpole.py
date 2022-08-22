import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gym
import numpy as np
from collections import deque

# discount factor for future utilities
DISCOUNT_FACTOR = 0.99
# number of episodes to run
NUM_EPISODES = 1000
# max steps per episode
MAX_STEPS = 10000
# score agent needs for environment to be solved
SOLVED_SCORE = 195
# device to run model on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARN_RATE = 0.001


# Using a neural network to learn our policy parameters
class PolicyNetwork(nn.Module):

  # Takes in observations and outputs actions
  def __init__(self, observation_space, action_space):
    super(PolicyNetwork, self).__init__()
    self.input_layer = nn.Linear(observation_space, 128)
    self.output_layer = nn.Linear(128, action_space)

  # forward pass
  def forward(self, x):
    # input states
    x = self.input_layer(x)

    # relu activation
    x = F.relu(x)

    # actions
    actions = self.output_layer(x)

    # get softmax for a probability distribution
    action_probs = F.softmax(actions, dim=1)

    return action_probs

  def select_action(self, state):
    """ Selects an action given current state
    Args:
    - network (Torch NN): network to process state
    - state (Array): Array of action space in an environment

    Return:
    - (int): action that is selected
    - (float): log probability of selecting that action given state and network
    """

    # make torch tensor of shape [BATCH x observation_size]
    state = torch.from_numpy(state).view(1, -1).to(DEVICE)

    # use network to predict action probabilities
    action_probs = self(state)
    state = state.detach()

    # sample an action using the probability distribution
    m = Categorical(action_probs)
    action = m.sample()

    # return action
    return action.item(), m.log_prob(action)


# Using a neural network to learn state value
class StateValueNetwork(nn.Module):

  # Takes in state
  def __init__(self, observation_space):
    super(StateValueNetwork, self).__init__()

    self.input_layer = nn.Linear(observation_space, 128)
    self.output_layer = nn.Linear(128, 1)

  # Expects X in shape [BATCH x observation_space]
  def forward(self, x):
    # input layer
    x = self.input_layer(x)

    # activiation relu
    x = F.relu(x)

    # get state value
    state_value = self.output_layer(x)

    return state_value


# Make environment
env = gym.make('CartPole-v1')

# Init network
print(f"Observation space: {env.observation_space.shape[0]}")
print(f"Action space: {env.action_space.n}")
policy_network = PolicyNetwork(env.observation_space.shape[0],
                               env.action_space.n).to(DEVICE)
stateval_network = StateValueNetwork(env.observation_space.shape[0]).to(DEVICE)

# Init optimizer
policy_optimizer = optim.Adam(policy_network.parameters(), lr=LEARN_RATE)
stateval_optimizer = optim.Adam(stateval_network.parameters(), lr=LEARN_RATE)

# track scores
scores = []

# track recent scores
recent_scores = deque(maxlen=100)

# run episodes
for episode in range(NUM_EPISODES):

  # init variables
  state = env.reset()
  done = False
  score = 0
  I = 1

  # run episode, update online
  for step in range(MAX_STEPS):
    env.render()
    # get action and log probability
    action, lp = policy_network.select_action(state)

    # step with action
    new_state, reward, done, _ = env.step(action)

    # update episode score
    score += reward

    # convert to torch tensor [Batch x observationsize]
    state_tensor = torch.from_numpy(state).reshape(1, -1).to(DEVICE)
    state_val = stateval_network(state_tensor)

    # get state value of next state
    new_state_tensor = torch.from_numpy(new_state).view(1, -1).to(DEVICE)
    new_state_val = stateval_network(new_state_tensor)

    # if terminal state, next state val is 0
    if done:
      print(f"Episode {episode} finished after {step} timesteps")
      new_state_val = torch.tensor([[0]]).to(DEVICE)

    # calculate value function loss with MSE
    val_loss = F.mse_loss(reward + DISCOUNT_FACTOR * new_state_val, state_val)
    # val_loss *= I

    # calculate policy loss
    advantage = reward + DISCOUNT_FACTOR * new_state_val.item() - state_val.item()
    # lp is tensor of shape [1], advantage is scalar
    policy_loss = -lp * advantage
    policy_loss *= I

    # Backpropagate policy
    policy_optimizer.zero_grad()
    # policy_loss.backward(retain_graph=True)
    policy_loss.backward()
    policy_optimizer.step()

    # Backpropagate value
    stateval_optimizer.zero_grad()
    val_loss.backward()
    stateval_optimizer.step()

    if done:
      break

    # move into new state, discount I
    state = new_state
    I *= DISCOUNT_FACTOR

  # append episode score
  scores.append(score)
  recent_scores.append(score)

  # early stopping if we meet solved score goal
  if np.array(recent_scores).mean() >= SOLVED_SCORE:
    break

np.savetxt('outputs/scores.txt', scores, delimiter=',')
