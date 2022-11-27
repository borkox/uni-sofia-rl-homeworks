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

# discount factor for future utilities
GAMA = 0.8
# number of episodes to run
NUM_EPISODES = 1000
# max steps per episode
MAX_STEPS = 10000
# score agent needs for environment to be solved
SOLVED_SCORE = 195
# device to run model on
time = 0
STEP = 15
REST_TIME = 50
scaler = scp.MinMaxScaler(feature_range=(0.01, 1), copy=True, clip=True)
# See https://www.gymlibrary.dev/environments/classic_control/cart_pole/#observation-space
scaler.fit([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [3.14, 5., 5., 5., 3.14, 5., 3.14, 5., 5., 3.14, 5., 3.14, 5., 5., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             3.14, 5., 5., 5., 3.14, 5., 3.14, 5., 5., 3.14, 5., 3.14, 5., 5., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             ]])


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
        abs(s[3]) if s[3] < 0 else 0,
        abs(s[4]) if s[4] > 0 else 0,
        abs(s[4]) if s[4] < 0 else 0,
        abs(s[5]) if s[5] > 0 else 0,
        abs(s[5]) if s[5] < 0 else 0,
        abs(s[6]) if s[6] > 0 else 0,
        abs(s[6]) if s[6] < 0 else 0,
        abs(s[7]) if s[7] > 0 else 0,
        abs(s[7]) if s[7] < 0 else 0,
        abs(s[8]) if s[8] > 0 else 0,
        abs(s[8]) if s[8] < 0 else 0,
        abs(s[9]) if s[9] > 0 else 0,
        abs(s[9]) if s[9] < 0 else 0,
        abs(s[10]) if s[10] > 0 else 0,
        abs(s[10]) if s[10] < 0 else 0,
        abs(s[11]) if s[11] > 0 else 0,
        abs(s[11]) if s[11] < 0 else 0,
        abs(s[12]) if s[12] > 0 else 0,
        abs(s[12]) if s[12] < 0 else 0,
        abs(s[13]) if s[13] > 0 else 0,
        abs(s[13]) if s[13] < 0 else 0,
        abs(s[14]) if s[14] > 0 else 0,
        abs(s[14]) if s[14] < 0 else 0,
        abs(s[15]) if s[15] > 0 else 0,
        abs(s[15]) if s[15] < 0 else 0,
        abs(s[16]) if s[16] > 0 else 0,
        abs(s[16]) if s[16] < 0 else 0,
        abs(s[17]) if s[17] > 0 else 0,
        abs(s[17]) if s[17] < 0 else 0,
        abs(s[18]) if s[18] > 0 else 0,
        abs(s[18]) if s[18] < 0 else 0,
        abs(s[19]) if s[19] > 0 else 0,
        abs(s[19]) if s[19] < 0 else 0,
        abs(s[20]) if s[20] > 0 else 0,
        abs(s[20]) if s[20] < 0 else 0,
        abs(s[21]) if s[21] > 0 else 0,
        abs(s[21]) if s[21] < 0 else 0,
        abs(s[22]) if s[22] > 0 else 0,
        abs(s[22]) if s[22] < 0 else 0,
        abs(s[23]) if s[23] > 0 else 0,
        abs(s[23]) if s[23] < 0 else 0])
    # print("Converting state: ", s, " ==> ", transformed)
    return transformed


# ================================================

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
dc_generator_env = nest.Create('dc_generator', 48, {"amplitude": 0.})

nest.Connect(dc_generator_env, INPUT,
             conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.5},
             syn_spec={'weight': 20})

nest.Connect(dc_generator_reward, SNc, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
             syn_spec={"weight": 50., "delay": 1.})


# =============================================================
def connect_WTA(INP):
    num_neurons = 50
    noise_weights = 20
    ex_weights = 10.5
    inh_weights = -2.6
    ex_inh_weights = 2.8
    action_1 = nest.Create("iaf_psc_alpha", num_neurons, {'I_e': 376.})
    action_2 = nest.Create("iaf_psc_alpha", num_neurons, {'I_e': 376.})
    action_3 = nest.Create("iaf_psc_alpha", num_neurons, {'I_e': 376.})
    action_4 = nest.Create("iaf_psc_alpha", num_neurons, {'I_e': 376.})
    wta_inhibitory = nest.Create("iaf_psc_alpha", num_neurons)
    all_actor_neurons = action_1 + action_2 + action_3 + action_4
    # n_input = nest.Create("poisson_generator", 10, {'rate': 3000.0})
    nest.Connect(INP, action_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 15},
                 syn_spec={'weight': noise_weights})
    nest.Connect(INP, action_2, conn_spec={'rule': 'fixed_indegree', 'indegree': 15},
                 syn_spec={'weight': noise_weights})
    nest.Connect(INP, action_3, conn_spec={'rule': 'fixed_indegree', 'indegree': 15},
                 syn_spec={'weight': noise_weights})
    nest.Connect(INP, action_4, conn_spec={'rule': 'fixed_indegree', 'indegree': 15},
                 syn_spec={'weight': noise_weights})
    nest.Connect(INP, wta_inhibitory, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
                 syn_spec={'weight': noise_weights * 0.9})
    nest.Connect(action_1, action_1, 'all_to_all', {'weight': ex_weights})
    nest.Connect(action_2, action_2, 'all_to_all', {'weight': ex_weights})
    nest.Connect(action_3, action_3, 'all_to_all', {'weight': ex_weights})
    nest.Connect(action_4, action_4, 'all_to_all', {'weight': ex_weights})
    nest.Connect(all_actor_neurons, wta_inhibitory, 'all_to_all', {'weight': ex_inh_weights})
    nest.Connect(wta_inhibitory, all_actor_neurons, 'all_to_all', {'weight': inh_weights})
    # sd = nest.Create("spike_recorder", 1)
    spike_recorder_1 = nest.Create("spike_recorder", 1)
    spike_recorder_2 = nest.Create("spike_recorder", 1)
    spike_recorder_3 = nest.Create("spike_recorder", 1)
    spike_recorder_4 = nest.Create("spike_recorder", 1)
    # spike_recorder_both = nest.Create("spike_recorder", 1)
    nest.Connect(action_1, spike_recorder_1, 'all_to_all')
    nest.Connect(action_2, spike_recorder_2, 'all_to_all')
    nest.Connect(action_3, spike_recorder_3, 'all_to_all')
    nest.Connect(action_4, spike_recorder_4, 'all_to_all')
    # nest.Connect(all_actor_neurons, spike_recorder_both, 'all_to_all')
    seed = np.random.randint(0, 1000000)
    nest.SetKernelStatus({'rng_seed': seed})
    return spike_recorder_1, spike_recorder_2, spike_recorder_3, spike_recorder_4


spike_recorders_1_4 = connect_WTA(SC)
spike_recorders_5_8 = connect_WTA(SC)


def motor(spike_count_arr):
    max_index = np.argmax(spike_count_arr)
    return np.linspace(-1, 1, len(spike_count_arr))[max_index]


def count_spikes(arr_spike_counters, time):
    result = []
    for sc in arr_spike_counters:
        result.append(
            len([e for e in nest.GetStatus(sc, keys='events')[0]['times'] if e > time])
        )
        # calc the "firerate" of each actor population
    return result


# =============================================================

# Make environment
env = gym.make('BipedalWalker-v2')

# Init network
print(f"Observation space: {env.observation_space.shape[0]}")
print(f"Action space: {env.action_space}")

# track scores
scores = []

# track recent scores
recent_scores = deque(maxlen=100)
prev_spikes = 0
# run episodes
for episode in range(NUM_EPISODES):

    # Clean spike count
    for x in spike_recorders_5_8:
        nest.SetStatus(x, {"n_events": 0})
    for x in spike_recorders_1_4:
        nest.SetStatus(x, {"n_events": 0})

    nest.SetStatus(spd_INPUT +
                       spd_SC+spd_D1 + spd_D2 +
                       spd_GPe + spd_STN +
                       spd_SNr + spd_SNc, {"n_events": 0})

    # init variables
    state = env.reset()
    print("STATE:+>>>>", state)
    done = False
    score = 0
    reward = 0
    new_reward = 0
    step = 0
    action = np.array([0.5, 0.5, 0.5, 0.5])
    sum_reward = 0
    # run episode, update online
    for _ in range(MAX_STEPS):

        # REWARD
        #     print("state: ", state)
        new_reward = max(reward,0) * 1000  # max(10 * math.cos(17 * state[2]), 0)

        print("New reward : ", new_reward)
        amplitude_I_reward = new_reward
        print("Setting amplitude for reward: ", amplitude_I_reward, "     step: ", step)
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

        spike_count_1_4 = count_spikes(spike_recorders_1_4, time)
        spike_count_5_8 = count_spikes(spike_recorders_1_4, time)

        action = np.array([
            motor(spike_count_1_4),
            motor(spike_count_5_8),
            -motor(spike_count_1_4),
            -motor(spike_count_5_8)])


        time += STEP
        time += REST_TIME
        print("actor spikes:", spike_count_1_4, spike_count_5_8, " at step ", step)

        # action = scaler_actions.transform(np.array([spikes_1,spikes_2,spikes_3,spikes_4]).reshape(1, -1)).reshape(-1)

        print("Action:", action)

        new_state, reward, done, _ = env.step(action)
        sum_reward += reward

        mean_reward = sum_reward / (step + 1)
        print("Mean reward: ", mean_reward)
        if mean_reward < -0 and step > 30:
            print("Early end episode for mean reward: ", mean_reward)
            done = True

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
# nest.raster_plot.from_device(spike_recorder_both, hist=True, title="ACTIONS")
# plt.show()
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
