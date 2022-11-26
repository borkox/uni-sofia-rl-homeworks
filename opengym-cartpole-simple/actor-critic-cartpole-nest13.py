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

seed = np.random.randint(0, 1000000)
nest.SetKernelStatus({'rng_seed': seed})
nest.SetKernelStatus({"resolution": 0.1})

# discount factor for future utilities
GAMA = 0.8
# number of episodes to run
NUM_EPISODES = 500
# max steps per episode
MAX_STEPS = 10000
# score agent needs for environment to be solved
SOLVED_SCORE = 195
# device to run model on
time = 0
STEP = 40
REST_TIME = 50
scaler = scp.MinMaxScaler(feature_range=(0.01, 1), copy=True, clip=True)
# See https://www.gymlibrary.dev/environments/classic_control/cart_pole/#observation-space
scaler.fit([[0, 0, 0, 0, 0, 0, 0, 0], [+1.5, +1.5, +1.5, +1.5, +0.13, +0.13, +2.1, +2.1]])

# ================================================
nest.set_verbosity("M_WARNING")
nest.ResetKernel()

def rand_w(w, percent):
    print('min=',w-abs(w*percent))
    print('max=',w + abs(w*percent))
    return nest.random.uniform(min=w-abs(w*percent), max=w + abs(w*percent))
# *****************************

## BG
BG_Nl=40
BG_Nr=40

bg_params={"V_th": -69.992,"V_reset": -80.0,"I_e": 100.0}

## Cortex
INPUT = nest.Create("iaf_psc_alpha", BG_Nl, bg_params)

## Striatum
D1 = nest.Create("iaf_psc_alpha", BG_Nl, bg_params)
D2 = nest.Create("iaf_psc_alpha", BG_Nl, bg_params)

## Actor
GPe = nest.Create("iaf_psc_alpha", BG_Nl, {'I_e': 380.})
STN = nest.Create("iaf_psc_alpha", BG_Nl, bg_params)
SNr = nest.Create("iaf_psc_alpha", BG_Nl, bg_params)
SNc = nest.Create("iaf_psc_alpha", BG_Nl, bg_params)

# reinforcement signal from object
nest.SetStatus(SNc, {'I_e': 200.0})

## State from object
nest.SetStatus(INPUT, {'I_e': 150.0})

## control actions
SC = nest.Create("iaf_psc_alpha", BG_Nl, bg_params)

## dopamine synapses
SNc_vt = nest.Create('volume_transmitter')
nest.Connect(SNc, SNc_vt,'all_to_all')

nest.CopyModel('stdp_dopamine_synapse', 'dopsyn', {'vt': SNc_vt.get('global_id'),
                                                   'A_plus': 0.01, 'A_minus': 0.01,
                                                   'Wmin':-2000.0, 'Wmax':2000.0})

nest.CopyModel('stdp_dopamine_synapse', 'adopsyn', {'vt': SNc_vt.get('global_id'),
                                                    'A_plus': -0.01, 'A_minus': -0.015,
                                                    'Wmin':-2000.0, 'Wmax':2000.0})

## Model connections
# nest.Connect(INPUT, INPUT, 'all_to_all', syn_spec={'weight': 100.0, "delay": 1.0})

nest.Connect(INPUT, D1,
             conn_spec={'rule': 'fixed_indegree', 'indegree': 15},
             syn_spec={'weight': rand_w(100.0, 0.2), "delay": 1.0, 'synapse_model': 'dopsyn'})

nest.Connect(INPUT, D2,
             conn_spec={'rule': 'fixed_indegree', 'indegree': 15},
             syn_spec={'weight': rand_w(40.0, 0.2), "delay": 1.0, 'synapse_model': 'adopsyn'})

nest.Connect(D1, D1,
             conn_spec={'rule': 'fixed_indegree', 'indegree': 15},
             syn_spec={'weight': 1.0, "delay": 1.0})
nest.Connect(D2, D2, "all_to_all", syn_spec={'weight': 1.0, "delay": 1.0})

nest.Connect(D2, GPe,
             conn_spec={'rule': 'fixed_indegree', 'indegree': 15},
             syn_spec={'weight': rand_w(-1.2, 0.2), "delay": 1.0}) #-10000.0

Gama=0.9
# Value function V(t)
nest.Connect(D1, SNc, 'one_to_one', syn_spec={'weight': rand_w(-1000.0, 0.2), "delay": 1.0, 'synapse_model': 'dopsyn'})
# Value function V(t+1)
nest.Connect(D1, SNc, 'one_to_one', syn_spec={'weight': rand_w(Gama*1000.0, 0.2), "delay": 2.0, 'synapse_model': 'dopsyn'})

# wrand=-100.0*random.random()
nest.Connect(D1, SNr, 'one_to_one',syn_spec={'weight': rand_w(-100.0, 0.2), "delay": 1.0, 'synapse_model': 'dopsyn'})

nest.Connect(GPe, STN, 'one_to_one', syn_spec={'weight': -100.0, "delay": 1.0}) #100.0!

nest.Connect(STN, GPe, 'one_to_one', syn_spec={'weight': 100.0, "delay": 1.0}) #100.0!

nest.Connect(STN, SNr, 'all_to_all', syn_spec={'weight': rand_w(100.0, 0.2), "delay": 1.0, 'synapse_model': 'dopsyn'})

g_SNr_SC=1.0
nest.Connect(SNr, SC, syn_spec={'weight': rand_w(g_SNr_SC, 0.2), "delay": 1.0})

g_INPUT=0.25
nest.Connect(INPUT, SC, syn_spec={'weight': rand_w(g_INPUT, 0.2), "delay": 1.0})

w_SC_D=100.0
nest.Connect(SC, D1,
             conn_spec={'rule': 'fixed_indegree', 'indegree': 15},
             syn_spec={'weight': rand_w(w_SC_D, 0.2), "delay": 1.0})
nest.Connect(SC, D2,
             conn_spec={'rule': 'fixed_indegree', 'indegree': 15},
             syn_spec={'weight': rand_w(w_SC_D, 0.2), "delay": 1.0})

gSC_INPUT=0.2
nest.Connect(SC, INPUT, syn_spec={'weight': gSC_INPUT, "delay": 1.0}) ###

spd_SC= nest.Create("spike_recorder")

spd_D1= nest.Create("spike_recorder")

spd_D2= nest.Create("spike_recorder")

spd_GPe= nest.Create("spike_recorder")

spd_STN= nest.Create("spike_recorder")

spd_SNr= nest.Create("spike_recorder")

spd_SNc= nest.Create("spike_recorder")
spd_INPUT= nest.Create("spike_recorder")

nest.Connect(SC,spd_SC)

nest.Connect(D1,spd_D1)
nest.Connect(D2,spd_D2)

nest.Connect(GPe,spd_GPe)

nest.Connect(STN,spd_STN)

nest.Connect(SNr,spd_SNr)

nest.Connect(SNc,spd_SNc)
nest.Connect(INPUT,spd_INPUT)

### end BG
# *****************************

dc_generator_reward = nest.Create('dc_generator', 8, {"amplitude": 0.})
dc_generator_env = nest.Create('dc_generator', 8, {"amplitude": 0.})

nest.Connect(dc_generator_env, INPUT,
             conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.5},
             syn_spec={'weight': 20})

nest.Connect(dc_generator_reward, SNc, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
             syn_spec={"weight": 50., "delay": 1.})


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


# =============================================================
num_neurons = 50
noise_weights = 25
ex_weights = 10.5
inh_weights = -2.6
ex_inh_weights = 2.8
action_left = nest.Create("iaf_psc_alpha", num_neurons)
action_right = nest.Create("iaf_psc_alpha", num_neurons)
wta_inhibitory = nest.Create("iaf_psc_alpha", num_neurons)
all_actor_neurons = action_left + action_right
# n_input = nest.Create("poisson_generator", 10, {'rate': 3000.0})
nest.Connect(SC, action_left, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
             syn_spec={'weight': noise_weights})
nest.Connect(SC, action_right, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
             syn_spec={'weight': noise_weights})
nest.Connect(SC, wta_inhibitory, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
             syn_spec={'weight': noise_weights * 0.9})
nest.Connect(action_left, action_left, 'all_to_all', {'weight': ex_weights})
nest.Connect(action_right, action_right, 'all_to_all', {'weight': ex_weights})
nest.Connect(all_actor_neurons, wta_inhibitory, 'all_to_all', {'weight': ex_inh_weights})
nest.Connect(wta_inhibitory, all_actor_neurons, 'all_to_all', {'weight': inh_weights})
# sd = nest.Create("spike_recorder", 1)
spike_recorder_l = nest.Create("spike_recorder", 1)
spike_recorder_r = nest.Create("spike_recorder", 1)
spike_recorder_both = nest.Create("spike_recorder", 1)
nest.Connect(action_left, spike_recorder_l, 'all_to_all')
nest.Connect(action_right, spike_recorder_r, 'all_to_all')
nest.Connect(all_actor_neurons, spike_recorder_both, 'all_to_all')
seed = np.random.randint(0, 1000000)
nest.SetKernelStatus({'rng_seed': seed})


# =============================================================

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

    nest.SetStatus(spd_INPUT +
                   spd_SC+spd_D1 + spd_D2 +
                   spd_GPe + spd_STN +
                   spd_SNr + spd_SNc +
                   spike_recorder_both +
                   spike_recorder_l +
                   spike_recorder_r, {"n_events": 0})

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

        # REWARD
        #     print("state: ", state)
        new_reward = max(10 * math.cos(17 * state[2]), 0)

        # print("New reward : ", new_reward)
        amplitude_I_reward = new_reward
        #     print("Setting amplitude for reward: ", amplitude_I_reward, "     step: ", step)
        # nest.SetStatus(dc_generator_reward, {"amplitude": amplitude_I_reward, "start": time, "stop": time + 25})
        nest.SetStatus(dc_generator_reward, {"amplitude": amplitude_I_reward})

        # ENVIRONMENT
        new_transformed_state = transform_state(state)
        #     print("new_transformed_state", new_transformed_state)
        new_transformed_state_scaled = scaler.transform(new_transformed_state.reshape(1, -1)).reshape(-1)
        #     print("new_transformed_state_scaled:", new_transformed_state_scaled)
        dc_environment_current = (np.exp(new_transformed_state_scaled) - 1) * 100.
        #     print("applying environment amplitude:", dc_environment_current)
        nest.SetStatus(dc_generator_env, {"amplitude": dc_environment_current})

        env.render()
        nest.Simulate(STEP)

        # REST for some time
        nest.SetStatus(dc_generator_env, {"amplitude": 0})
        nest.Simulate(REST_TIME)

        left_spikes = len([e for e in nest.GetStatus(spike_recorder_l, keys='events')[0]['times'] if
                           e > time])  # calc the "firerate" of each actor population
        right_spikes = len([e for e in nest.GetStatus(spike_recorder_r, keys='events')[0]['times'] if
                            e > time])  # calc the "firerate" of each actor population
        time += STEP
        time += REST_TIME
        print("actor spikes2:", left_spikes, right_spikes, " at step ", step)

        action = 0 if left_spikes > right_spikes else 1

        #     print("Action:", action)

        new_state, reward, done, _ = env.step(action)

        if done:
            for i in range(0, 5):
                step = step + 1
                nest.SetStatus(dc_generator_reward, {"amplitude": 0.})
                nest.Simulate(STEP + REST_TIME)
                time += STEP + REST_TIME

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



print("Save scores")
np.savetxt('outputs/scores.txt', scores, delimiter=',')



nest.raster_plot.from_device(spd_INPUT, hist=True, title="INPUT")
plt.show()
nest.raster_plot.from_device(spike_recorder_both, hist=True, title="ACTIONS")
plt.show()
nest.raster_plot.from_device(spd_SC, hist=True, title="spd_SC")
plt.show()
nest.raster_plot.from_device(spd_D1, hist=True, title="spd_D1")
plt.show()
nest.raster_plot.from_device(spd_D2, hist=True, title="spd_D2")
plt.show()
nest.raster_plot.from_device(spd_GPe, hist=True, title="spd_GPe")
plt.show()
nest.raster_plot.from_device(spd_STN, hist=True, title="spd_STN")
plt.show()
nest.raster_plot.from_device(spd_SNr, hist=True, title="spd_SNr")
plt.show()
nest.raster_plot.from_device(spd_SNc, hist=True, title="spd_SNc")
plt.show()

