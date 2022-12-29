import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
import gym
import numpy as np
from collections import deque
import math

# discount factor for future utilities
DISCOUNT_FACTOR = 0.98
# number of episodes to run
NUM_EPISODES = 1000
# max steps per episode
MAX_STEPS = 10000
# score agent needs for environment to be solved
SOLVED_SCORE = 1600
# device to run model on
DEVICE = "cpu"##"cuda" if torch.cuda.is_available() else "cpu"
LEARN_RATE = 0.00001
print(f"DEVICE = {DEVICE}")

def calc_logprob(mu_v, var_v, actions_v):
  p1 = - ((mu_v - torch.FloatTensor(actions_v)) ** 2) / (2*var_v.clamp(min=1e-3))
  p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
  return p1 + p2

# Using a neural network to learn our policy parameters
class PolicyNetwork(nn.Module):

  # Takes in observations and outputs actions
  def __init__(self, observation_space, action_space):
    super(PolicyNetwork, self).__init__()
    self.input_layer = nn.Linear(observation_space, 512)
    self.hidden_layer1 = nn.Linear(512, 512)
    self.hidden_layer2 = nn.Linear(512, 512)
    self.dropout = nn.Dropout(0.5)
    self.mu_layer = nn.Linear(512, action_space)
    self.sigma_layer = nn.Linear(512, action_space)

  # forward pass
  def forward(self, x):
    # input states
    x = self.input_layer(x)
    x = F.relu(x)
    x = self.dropout(self.hidden_layer1(x))
    x = F.relu(x)
    x = self.dropout(self.hidden_layer2(x))
    x = F.relu(x)
    # mu is vector of float in range [-1;1]
    mu = torch.tanh(self.mu_layer(x))
    var = F.softplus(self.sigma_layer(x))
    return mu, var

  def select_action(self, state):
    # make torch tensor of shape [BATCH x observation_size]
    state = torch.from_numpy(state).view(1, -1).to(DEVICE)

    # use network to predict action probabilities
    mu_v, var_v = self(state)

    mu = mu_v.data.cpu().numpy()
    sigma = torch.sqrt(var_v).data.cpu().numpy()
    actions = np.random.normal(mu, sigma)
    actions = np.clip(actions, -1, 1)

    return actions.reshape(-1), mu_v, var_v


# Using a neural network to learn state value
class StateValueNetwork(nn.Module):

  # Takes in state
  def __init__(self, observation_space):
    super(StateValueNetwork, self).__init__()

    self.input_layer = nn.Linear(observation_space, 512)
    self.mid_layer = nn.Linear(512, 512)
    self.output_layer = nn.Linear(512, 1)

  # Expects X in shape [BATCH x observation_space]
  def forward(self, x):
    # input layer
    x = self.input_layer(x)
    x = F.relu(x)
    x = F.dropout(x, 0.5)
    x = self.mid_layer(x)
    x = F.relu(x)
    state_value = self.output_layer(x)
    return state_value


# Make environment
env = gym.make('BipedalWalker-v2')

# Init network
print(f"Observation space: {env.observation_space.shape[0]}")
print(f"Action space: {env.action_space.shape[0]}")
policy_network = PolicyNetwork(env.observation_space.shape[0],
                               env.action_space.shape[0]).to(DEVICE)
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
    action, mu_v, var_v = policy_network.select_action(state)

    # step with action
    #print('action=', action)
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
      print(f"Episode {episode} finished after {step} timesteps, score={score}")
      new_state_val = torch.tensor([[0]]).to(DEVICE)

    # calculate value function loss with MSE
    val_loss = F.mse_loss(reward + DISCOUNT_FACTOR * new_state_val.reshape(-1,1), state_val.reshape(-1,1))
    val_loss *= I

    # calculate policy loss
    advantage = reward + DISCOUNT_FACTOR * new_state_val.item() - state_val.item()

    log_prob_v = advantage * calc_logprob(mu_v, var_v, action)
    loss_policy_v = -log_prob_v.mean()
    entropy_loss_v = 1e-4 * (-(torch.log(2*math.pi*var_v) + 1)/2).mean()

    loss_v = loss_policy_v + entropy_loss_v

    # Backpropagate policy
    policy_optimizer.zero_grad()
    # policy_loss.backward(retain_graph=True)
    #print(f"policy_loss={policy_loss}")
    loss_v.backward()
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
  if episode % 10 == 0:
    print("Saving ......")
    print(f"len(scores) = {len(scores)}")
    np.savetxt('outputs/scores.txt', scores, delimiter=',')

np.savetxt('outputs/scores.txt', scores, delimiter=',')
