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
NUM_EPISODES = 3000
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

nest.set_verbosity("M_WARNING")
nest.ResetKernel()

SNc_vt = nest.Create('volume_transmitter')

STATE = nest.Create("iaf_psc_alpha", 6 * 24, {"I_e": 130.0})
V = nest.Create("iaf_psc_alpha", 40, {"I_e": 30.0})
POLICY = nest.Create("iaf_psc_alpha", 300, {"I_e": 30.0})
# PD = nest.Create("iaf_psc_alpha", 15, {"I_e": 30.0})
SNc = nest.Create("iaf_psc_alpha", 8, {"I_e": 10.0})

dc_generator_reward = nest.Create('dc_generator', 8, {"amplitude": 0.})
dc_generator_env = nest.Create('dc_generator', 48, {"amplitude": 0.})
spike_recorder_STATE = nest.Create('spike_recorder')
spike_recorder_V = nest.Create('spike_recorder')
spike_recorder_POLICY = nest.Create('spike_recorder')
spike_recorder_SNc = nest.Create('spike_recorder')
nest.CopyModel('stdp_dopamine_synapse', 'dopsyn', \
               {'vt': SNc_vt.get('global_id'), \
                'A_plus': 0.01, 'A_minus': 0.01, \
                'Wmin': -3000.0, 'Wmax': 3000.0,
                # 'tau_c': 5 * (STEP + REST_TIME),
                # 'tau_n': 3 * (STEP + REST_TIME),
                'tau_plus': 20.0
                })

nest.Connect(dc_generator_env, STATE,
             conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.2},
             syn_spec={'weight': 20})

nest.Connect(STATE, STATE,
             conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.1},
             syn_spec={'weight': 1})

nest.Connect(STATE[3*24:5*24], V,
             conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.5},
             syn_spec={
                 "weight": nest.random.uniform(min=-20., max=45.),
                 'synapse_model': 'dopsyn'})
nest.Connect(STATE[5*24:], V,
             conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.5},
             syn_spec={
                 "weight": nest.random.uniform(min=-20., max=45.),
                 'synapse_model': 'dopsyn', "delay": STEP + REST_TIME + 1.0})

nest.Connect(STATE[0:2*24], POLICY,
             conn_spec={'rule': 'fixed_indegree', 'indegree': 15},
             syn_spec={
                 "weight": nest.random.uniform(min=-20., max=45.),
                 'synapse_model': 'dopsyn'})
nest.Connect(STATE[2*24:3*24], POLICY,
             conn_spec={'rule': 'fixed_indegree', 'indegree': 15},
             syn_spec={
                 "weight": nest.random.uniform(min=-20., max=45.),
                 'synapse_model': 'dopsyn', "delay": STEP + REST_TIME + 1.0})

# nest.Connect(POLICY, STATE,
#              conn_spec={'rule': 'fixed_indegree', 'indegree': 25},
#              syn_spec={
#                  "weight": nest.random.uniform(min=10., max=25.)})

nest.Connect(dc_generator_reward, SNc, 'one_to_one',
             syn_spec={"weight": 50., "delay": 1.})

# Volume transmitter
nest.Connect(SNc, SNc_vt, 'all_to_all')

# Value function V(t)
nest.Connect(V, SNc, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
             syn_spec={'weight': - 220.0, "delay": STEP + REST_TIME + 1.0})
# Value function V(t+1)
nest.Connect(V, SNc, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
             syn_spec={'weight': GAMA * 220.0, "delay": 1.})

nest.Connect(STATE, spike_recorder_STATE)
nest.Connect(V, spike_recorder_V)
nest.Connect(POLICY, spike_recorder_POLICY)
nest.Connect(SNc, spike_recorder_SNc)
# *****************************

dc_generator_reward = nest.Create('dc_generator', 8, {"amplitude": 0.})
dc_generator_env = nest.Create('dc_generator', 48, {"amplitude": 0.})

nest.Connect(dc_generator_env, STATE,
             conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.5},
             syn_spec={'weight': 20})

nest.Connect(dc_generator_reward, SNc, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
             syn_spec={"weight": 50., "delay": 1.})


# =============================================================
def connect_WTA(INP):
    num_neurons = 20
    noise_weights = 20
    ex_weights = 10.5
    inh_weights = -2.6
    ex_inh_weights = 2.8
    action_1 = nest.Create("iaf_psc_alpha", num_neurons, {'I_e': 376.})
    action_2 = nest.Create("iaf_psc_alpha", num_neurons, {'I_e': 376.})
    action_3 = nest.Create("iaf_psc_alpha", num_neurons, {'I_e': 376.})
    action_4 = nest.Create("iaf_psc_alpha", num_neurons, {'I_e': 376.})
    action_5 = nest.Create("iaf_psc_alpha", num_neurons, {'I_e': 376.})
    wta_inhibitory = nest.Create("iaf_psc_alpha", num_neurons)
    all_actor_neurons = action_1 + action_2 + action_3 + action_4 + action_5
    # n_input = nest.Create("poisson_generator", 10, {'rate': 3000.0})
    nest.Connect(INP, action_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 15},
                 syn_spec={'weight': noise_weights})
    nest.Connect(INP, action_2, conn_spec={'rule': 'fixed_indegree', 'indegree': 15},
                 syn_spec={'weight': noise_weights})
    nest.Connect(INP, action_3, conn_spec={'rule': 'fixed_indegree', 'indegree': 15},
                 syn_spec={'weight': noise_weights})
    nest.Connect(INP, action_4, conn_spec={'rule': 'fixed_indegree', 'indegree': 15},
                 syn_spec={'weight': noise_weights})
    nest.Connect(INP, action_5, conn_spec={'rule': 'fixed_indegree', 'indegree': 15},
                 syn_spec={'weight': noise_weights})
    nest.Connect(INP, wta_inhibitory, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
                 syn_spec={'weight': noise_weights * 0.9})
    nest.Connect(action_1, action_1, 'all_to_all', {'weight': ex_weights})
    nest.Connect(action_2, action_2, 'all_to_all', {'weight': ex_weights})
    nest.Connect(action_3, action_3, 'all_to_all', {'weight': ex_weights})
    nest.Connect(action_4, action_4, 'all_to_all', {'weight': ex_weights})
    nest.Connect(action_5, action_5, 'all_to_all', {'weight': ex_weights})
    nest.Connect(all_actor_neurons, wta_inhibitory, 'all_to_all', {'weight': ex_inh_weights})
    nest.Connect(wta_inhibitory, all_actor_neurons, 'all_to_all', {'weight': inh_weights})
    # sd = nest.Create("spike_recorder", 1)
    spike_recorder_1 = nest.Create("spike_recorder", 1)
    spike_recorder_2 = nest.Create("spike_recorder", 1)
    spike_recorder_3 = nest.Create("spike_recorder", 1)
    spike_recorder_4 = nest.Create("spike_recorder", 1)
    spike_recorder_5 = nest.Create("spike_recorder", 1)
    # spike_recorder_both = nest.Create("spike_recorder", 1)
    nest.Connect(action_1, spike_recorder_1, 'all_to_all')
    nest.Connect(action_2, spike_recorder_2, 'all_to_all')
    nest.Connect(action_3, spike_recorder_3, 'all_to_all')
    nest.Connect(action_4, spike_recorder_4, 'all_to_all')
    nest.Connect(action_5, spike_recorder_5, 'all_to_all')
    # nest.Connect(all_actor_neurons, spike_recorder_both, 'all_to_all')
    seed = np.random.randint(0, 1000000)
    nest.SetKernelStatus({'rng_seed': seed})
    return spike_recorder_1, spike_recorder_2, spike_recorder_3, spike_recorder_4, spike_recorder_5


spike_recorders_group_1 = connect_WTA(POLICY)
spike_recorders_group_2 = connect_WTA(POLICY)
spike_recorders_group_3 = connect_WTA(POLICY)
spike_recorders_group_4 = connect_WTA(POLICY)


def motor_action(spike_count_group_1, spike_count_group_2, spike_count_group_3, spike_count_group_4, prev_action):
    count = len(spike_count_group_1)
    lin_space_interval = np.linspace(-0.5, 0.5, count)
    action = np.clip(np.array([
        lin_space_interval[np.argmax(spike_count_group_1)],
        lin_space_interval[np.argmax(spike_count_group_2)],
        lin_space_interval[np.argmax(spike_count_group_3)],
        lin_space_interval[np.argmax(spike_count_group_4)]
    ]), -1, 1)
    return action


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
    for x in spike_recorders_group_1:
        nest.SetStatus(x, {"n_events": 0})
    for x in spike_recorders_group_2:
        nest.SetStatus(x, {"n_events": 0})
    for x in spike_recorders_group_3:
        nest.SetStatus(x, {"n_events": 0})
    for x in spike_recorders_group_4:
        nest.SetStatus(x, {"n_events": 0})

    nest.SetStatus(spike_recorder_STATE, {"n_events": 0})
    nest.SetStatus(spike_recorder_V, {"n_events": 0})
    nest.SetStatus(spike_recorder_POLICY, {"n_events": 0})
    nest.SetStatus(spike_recorder_SNc, {"n_events": 0})

    # init variables
    state = env.reset()
    print("STATE:+>>>>", state)
    done = False
    score = 0
    reward = 0
    new_reward = 0
    recent_reward_mean = 0
    step = 0
    action = np.array([0., 0., 0., 0.])
    recent_episode_rewards = deque(maxlen=30)

    # run episode, update online
    for _ in range(MAX_STEPS):

        # REWARD
        #     print("state: ", state)
        new_reward = max(recent_reward_mean+0.2,0) * 100  # max(10 * math.cos(17 * state[2]), 0)

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

        spike_count_group_1 = count_spikes(spike_recorders_group_1, time)
        spike_count_group_2 = count_spikes(spike_recorders_group_2, time)
        spike_count_group_3 = count_spikes(spike_recorders_group_3, time)
        spike_count_group_4 = count_spikes(spike_recorders_group_4, time)

        action = motor_action(spike_count_group_1,
                              spike_count_group_2,
                              spike_count_group_3,
                              spike_count_group_4,
                              action)


        time += STEP
        time += REST_TIME
        print("actor spikes:", spike_count_group_1, spike_count_group_2, spike_count_group_3, spike_count_group_4, " at step ", step)

        # action = scaler_actions.transform(np.array([spikes_1,spikes_2,spikes_3,spikes_4]).reshape(1, -1)).reshape(-1)

        print("Action:", action)

        new_state, reward, done, _ = env.step(action)

        print("Reward: ", reward)
        recent_episode_rewards.append(reward)

        recent_reward_mean = np.array(recent_episode_rewards).mean()
        print("Recent episode rewards mean: ", recent_reward_mean)
        if recent_reward_mean < -0 and step > 50:
            print("Early end episode for mean reward: ", recent_reward_mean)
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


print("====== V === SNc ===")
print(nest.GetConnections(V, SNc))
print("====== STATE === V ===")
print(nest.GetConnections(STATE, V))
print("====== STATE === POLICY ===")
print(nest.GetConnections(STATE, POLICY))

nest.raster_plot.from_device(spike_recorder_STATE, hist=True, title="STATE")
plt.show()
nest.raster_plot.from_device(spike_recorder_V, hist=True, title="V")
plt.show()
nest.raster_plot.from_device(spike_recorder_POLICY, hist=True, title="POLICY")
plt.show()
nest.raster_plot.from_device(spike_recorder_SNc, hist=True, title="SNc")
plt.show()