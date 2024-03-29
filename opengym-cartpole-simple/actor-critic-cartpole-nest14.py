import gym
import numpy as np
from collections import deque
import random
import math
import time
import sklearn.preprocessing as scp
import nest
import nest.voltage_trace
import matplotlib.pyplot as plt
import numpy as np

# number of episodes to run
NUM_EPISODES = 590
# max steps per episode
MAX_STEPS = 10000
# score agent needs for environment to be solved
SOLVED_SCORE = 195
# device to run model on
time = 0
STEP = 200
REST_TIME = 50
scaler = scp.MinMaxScaler(feature_range=(0.01, 0.99), copy=True, clip=True)
# See https://www.gymlibrary.dev/environments/classic_control/cart_pole/#observation-space
scaler.fit([[0, 0, 0, 0, 0, 0, 0, 0], [+1.5, +1.5, +1.5, +1.5, +0.13, +0.13, +2, +2]])

# ================================================
nest.set_verbosity("M_WARNING")
nest.ResetKernel()

# ================================================
world_dim = {'x': 8, 'y': 1}
num_actions = 2

# NUM_ITERATIONS = 300
NUM_STATE_NEURONS = 20
NUM_WTA_NEURONS = 50
WEIGHT_SCALING = 100 / NUM_STATE_NEURONS

# nest.ResetKernel()
# nest.set_verbosity("M_DEBUG")

rank = nest.Rank()
size = nest.NumProcesses()
seed = np.random.randint(0, 1000000)
num_threads = 1
nest.SetKernelStatus({"local_num_threads": num_threads})
nest.SetKernelStatus({"rng_seed": seed})
tau_pre = 20.
nest.SetDefaults("iaf_psc_alpha", {"tau_minus": tau_pre})

# Create states
states = []
all_states = None
for i in range(world_dim['x']):
    states.append([])
    state_group = nest.Create('iaf_psc_alpha', NUM_STATE_NEURONS)
    for j in range(world_dim['y']):
        states[i].append(state_group)
    if all_states is None:
        all_states = state_group
    else:
        all_states = all_states + state_group

# Create actions
actions = []
all_actions = None
for i in range(num_actions):
    action_group = nest.Create('iaf_psc_alpha', NUM_WTA_NEURONS)
    actions.append(action_group)
    if all_actions is None:
        all_actions = action_group
    else:
        all_actions = all_actions + action_group


# Create WTA circuit
wta_ex_weights = 10.5
wta_inh_weights = -2.6
wta_ex_inh_weights = 2.8
wta_noise_weights = 2.1

wta_inh_neurons = nest.Create('iaf_psc_alpha', NUM_WTA_NEURONS)

for i in range(len(actions)):
    nest.Connect(actions[i], actions[i], 'all_to_all', {'weight': wta_ex_weights})
    nest.Connect(actions[i], wta_inh_neurons, 'all_to_all', {'weight': wta_ex_inh_weights})

nest.Connect(wta_inh_neurons, all_actions, 'all_to_all', {'weight': wta_inh_weights})

wta_noise = nest.Create('poisson_generator', 10, {'rate': 3000.})
nest.Connect(wta_noise, all_actions, 'all_to_all', {'weight': wta_noise_weights})
nest.Connect(wta_noise, wta_inh_neurons, 'all_to_all', {'weight': wta_noise_weights * 0.9})


# Create stimulus
stimulus = nest.Create('poisson_generator', 1, {'rate': 5000.})
nest.Connect(stimulus, all_states, 'all_to_all', {'weight': 0.})


# Here, we are implementing the dopaminergic nueron pool, volume transmitter and dopamin-modulated synapse between states and actions

# Create DA pool
DA_neurons = nest.Create('iaf_psc_alpha', 100)
vol_trans = nest.Create('volume_transmitter', 1, {'deliver_interval': 10})
nest.Connect(DA_neurons, vol_trans, 'all_to_all')

# Create reward stimulus
reward_stimulus = nest.Create('poisson_generator', 1, {'rate': 5000.})
nest.Connect(reward_stimulus, DA_neurons, 'all_to_all', {'weight': 0.})

tau_c = 50.0
tau_n = 20.0
tau_post = 20.

# Connect states to actions
nest.CopyModel('stdp_dopamine_synapse', 'dopa_synapse', {
    'vt': vol_trans.get('global_id'), 'A_plus': 0.040, 'A_minus': 0.050, "tau_plus": tau_post,
    'Wmin': -10., 'Wmax': 10., 'b': 1., 'tau_n': tau_n, 'tau_c': tau_c})

nest.Connect(all_states, all_actions, 'all_to_all', {'synapse_model': 'dopa_synapse', 'weight': 0.0})

# TODO experimental: project from state to DA via critic 
nest.CopyModel('stdp_dopamine_synapse', 'dopa_synapse_critic', {
    'vt': vol_trans.get('global_id'), 'A_plus': 0.040, 'A_minus': 0.050, "tau_plus": tau_post,
    'Wmin': -10., 'Wmax': 10., 'b': 1., 'tau_n': tau_n, 'tau_c': tau_c})

critic = nest.Create('iaf_psc_alpha', 50)
nest.Connect(all_states, critic, 'all_to_all', {'synapse_model': 'dopa_synapse_critic', 'weight': 0.0})
nest.Connect(critic, DA_neurons, 'all_to_all', {'weight': -5., 'delay': 50.})
nest.Connect(critic, DA_neurons, 'all_to_all', {'weight': 5., 'delay': 1.})

critic_noise = nest.Create('poisson_generator', 1, {'rate': 65500.})
nest.Connect(critic_noise, critic)

# Create spike detector
sd_wta = nest.Create('spike_recorder')
nest.Connect(all_actions, sd_wta)
nest.Connect(wta_inh_neurons, sd_wta)
sd_actions = nest.Create('spike_recorder', num_actions)
for i in range(len(actions)):
    nest.Connect(actions[i], sd_actions[i])
sd_states = nest.Create('spike_recorder')
nest.Connect(all_states, sd_states)
sd_DA = nest.Create('spike_recorder', 1)
nest.Connect(DA_neurons, sd_DA, 'all_to_all')
sd_critic = nest.Create('spike_recorder', 1)
nest.Connect(critic, sd_critic, 'all_to_all')


# Create noise
noise = nest.Create('poisson_generator', 1, {'rate': 65000.})
nest.Connect(noise, all_states, 'all_to_all', {'weight': 1.})
nest.Connect(noise, DA_neurons, 'all_to_all', {'weight': 1.0367})


# ================================================

def transform_state(s):
    transformed = np.array([
        abs(s[0]) if s[0] > 0 else 0,
        abs(s[0]) if s[0] < 0 else 0,
        abs(s[1]) if s[1] > 0 else 0,
        abs(s[1]) if s[1] < 0 else 0,
        abs(s[2]) if s[2] > 0 else 0,
        abs(s[2]) if s[2] < 0 else 0,
        abs(s[3]) if s[3] > 0 else 0,
        abs(s[3]) if s[3] < 0 else 0])
    # print("Converting state: ", s, " ==> ", transformed)
    return transformed

# Make environment
env = gym.make('CartPole-v1')

# Init network
print(f"Observation space: {env.observation_space.shape[0]}")
print(f"Action space: {env.action_space.n}")

# track scores
scores = []

# track recent scores
recent_scores = deque(maxlen=100)
prev_spikes = 0
# run episodes
for episode in range(NUM_EPISODES):

    nest.SetStatus(sd_actions, {"n_events": 0})
    nest.SetStatus(sd_wta, {"n_events": 0})
    nest.SetStatus(sd_states, {"n_events": 0})
    nest.SetStatus(sd_DA, {"n_events": 0})
    nest.SetStatus(sd_critic, {"n_events": 0})

    # init variables
    state = env.reset()
    print("STATE:+>>>>", state)
    done = False
    score = 0
    reward = 0
    new_reward = 0
    step = 0
    # run episode, update online
    for _ in range(MAX_STEPS):

        nest.SetStatus(wta_noise, {'rate': 3000.})

        # ENVIRONMENT
        new_transformed_state = transform_state(state)
        #     print("new_transformed_state", new_transformed_state)
        new_transformed_state_scaled = scaler.transform(new_transformed_state.reshape(1, -1)).reshape(-1)
        #     print("new_transformed_state_scaled:", new_transformed_state_scaled)
        # dc_environment_current = (np.exp(new_transformed_state_scaled) - 1) * 100.
        #     print("applying environment amplitude:", dc_environment_current)
        # nest.SetStatus(dc_generator_env, {"amplitude": dc_environment_current})
        for si in range(len(states)):
            nest.SetStatus(nest.GetConnections(stimulus, states[si][0]), {'weight': new_transformed_state_scaled[si]})

        env.render()
        nest.Simulate(STEP)

        max_rate = -1
        chosen_action = -1
        for i in range(len(sd_actions)):
            rate = len([e for e in nest.GetStatus(sd_actions[i], keys='events')[0]['times'] if e > time]) # calc the \"firerate\" of each actor population
            if rate > max_rate:
                max_rate = rate # the population with the hightes rate wins
                chosen_action = i
        possible_actions = [0,1]

        time += STEP
        print("chose action:", possible_actions[chosen_action] , " at step ", step)

        action = possible_actions[chosen_action]
        new_state, reward, done, _ = env.step(action)

        # stimulate new state
        for si in range(len(states)):
            nest.SetStatus(nest.GetConnections(stimulus, states[si][0]), {'weight': 0.})

        # apply reward
        new_reward = max(math.cos(17 * new_state[2]), 0)
        nest.SetStatus(nest.GetConnections(reward_stimulus, DA_neurons), {'weight': float(new_reward)* WEIGHT_SCALING})
        nest.SetStatus(wta_noise, {'rate': 0.})

        # refactory time
        nest.Simulate(REST_TIME)
        time += REST_TIME

        nest.SetStatus(nest.GetConnections(reward_stimulus, DA_neurons), {'weight': 0.0})

        # if done:
        #     for i in range(0, 1):
        #         step = step + 1
        #         nest.SetStatus(dc_generator_reward, {"amplitude": -10.})
        #         nest.Simulate(STEP + REST_TIME)
        #         time += STEP + REST_TIME

        #     print("reward:", reward)

        # update episode score
        score += reward

        # if terminal state, next state val is 0
        if done:
            print(f"Episode {episode} finished after {step} timesteps")
            break

        # move into new state, discount I
        state = new_state
        step = step + 1

    # append episode score
    scores.append(score)
    recent_scores.append(score)

    # early stopping if we meet solved score goal
    if np.array(recent_scores).mean() >= SOLVED_SCORE:
        break
    if len(scores) % 10 == 0:
        print("Save scores")
        np.savetxt('outputs/scores.txt', scores, delimiter=',')

np.savetxt('outputs/scores.txt', scores, delimiter=',')

print("====== all_states === all_actions ===")
print(nest.GetConnections(all_states, all_actions))

nest.raster_plot.from_device(sd_wta, hist=True, title="sd_wta")
plt.show()
nest.raster_plot.from_device(sd_states, hist=True, title="sd_states")
plt.show()
nest.raster_plot.from_device(sd_DA, hist=True, title="sd_DA")
plt.show()
nest.raster_plot.from_device(sd_critic, hist=True, title="sd_critic")
plt.show()
