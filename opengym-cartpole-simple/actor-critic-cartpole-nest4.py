
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

# discount factor for future utilities
GAMA = 0.9
# number of episodes to run
NUM_EPISODES = 5
# max steps per episode
MAX_STEPS = 10000
# score agent needs for environment to be solved
SOLVED_SCORE = 195
# device to run model on

STEP = 100
scaler = scp.MinMaxScaler(feature_range=(0, 1), copy=True, clip=True)
# See https://www.gymlibrary.dev/environments/classic_control/cart_pole/#observation-space
scaler.fit([[0,0,0,0,0,0,0,0],[+0.5,+0.5,+1.5,+1.5,+0.3,+0.3,+3,+3]])

#================================================
nest.set_verbosity("M_WARNING")
nest.ResetKernel()


SNc_vt = nest.Create('volume_transmitter')

STATE = nest.Create("iaf_psc_alpha", 50, {"I_e": 10.0})
V      = nest.Create("iaf_psc_alpha", 50, {"I_e": 30.0})
# G      = nest.Create("iaf_psc_alpha", 100, {"I_e": 30.0})
SNc     = nest.Create("iaf_psc_alpha", 8, {"I_e": 10.0})
# SNr_L     = nest.Create("iaf_psc_alpha", 40, {"I_e": 50.0})
# SNr_R     = nest.Create("iaf_psc_alpha", 40, {"I_e": 50.0})
ACTION_L  = nest.Create("iaf_psc_alpha", 10, {"I_e": 0.0})
ACTION_R  = nest.Create("iaf_psc_alpha", 10, {"I_e": 0.0})


dc_generator_reward = nest.Create('dc_generator', 8, {"amplitude": 0.})
dc_generator_env = nest.Create('dc_generator', 8, {"amplitude": 0.})
# noise_generator = nest.Create('poisson_generator', 4, {"rate": 50.})
# voltmeter_STATE = nest.Create("voltmeter")
# voltmeter_V = nest.Create("voltmeter")
# voltmeter_SNc = nest.Create("voltmeter")
voltmeter_ACTION_L = nest.Create("voltmeter")
voltmeter_ACTION_R = nest.Create("voltmeter")
spike_recorder_STATE = nest.Create('spike_recorder')
spike_recorder_V = nest.Create('spike_recorder')
# spike_recorder_G = nest.Create('spike_recorder')
spike_recorder_SNc = nest.Create('spike_recorder')
# spike_recorder_SNr_L = nest.Create('spike_recorder')
# spike_recorder_SNr_R = nest.Create('spike_recorder')
spike_recorder_ACTION_L = nest.Create('spike_recorder')
spike_recorder_ACTION_R = nest.Create('spike_recorder')
nest.CopyModel('stdp_dopamine_synapse', 'dopsyn', \
               { 'vt': SNc_vt.get('global_id'), \
                'A_plus': 0.001, 'A_minus': 0.001, \
                'Wmin': -30000.0, 'Wmax':30000.0})

nest.Connect(dc_generator_env, STATE,
             conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.5},
            syn_spec={'weight': 20 })

nest.Connect(STATE, V, \
            conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.5},
            syn_spec={
            "weight": nest.random.uniform(min= -20., max=45.),
            'synapse_model': 'dopsyn'})

nest.Connect(dc_generator_reward, SNc,'one_to_one', \
             syn_spec={ "weight": 50.})

# Volume transmitter
nest.Connect(SNc, SNc_vt,'all_to_all')


# Value function V(t)
nest.Connect(V, SNc, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
             syn_spec={'weight': -200.0, "delay": 1.0})
# Value function V(t+1)
nest.Connect(V, SNc, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
             syn_spec={'weight': GAMA*200.0, "delay": 2.})

nest.Connect(V, ACTION_L, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
             syn_spec={'weight': nest.random.uniform(min= -10., max=85.)})
nest.Connect(V, ACTION_R, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
             syn_spec={'weight': nest.random.uniform(min= -10., max=85.)})


# nest.Connect(SNr_L, ACTION_L,
#             syn_spec={'weight': 15.0 })
# nest.Connect(SNr_R, ACTION_R,
#             syn_spec={'weight': 15.0 })

nest.Connect(ACTION_L, ACTION_R, 'all_to_all', syn_spec={'weight': -50.0})
nest.Connect(ACTION_R, ACTION_L, 'all_to_all', syn_spec={'weight': -50.0})


nest.Connect(STATE, spike_recorder_STATE)
nest.Connect(V, spike_recorder_V)
# nest.Connect(G, spike_recorder_G)
nest.Connect(SNc, spike_recorder_SNc)
# nest.Connect(SNr_L, spike_recorder_SNr_L)
# nest.Connect(SNr_R, spike_recorder_SNr_R)
nest.Connect(ACTION_L, spike_recorder_ACTION_L)
nest.Connect(ACTION_R, spike_recorder_ACTION_R)

#================================================
def transform_state(s):
  transformed = np.array([
       abs(s[0]) if s[0] > 0 else 0,
       abs(s[0]) if s[0] < 0 else 0,
       abs(s[1]) if s[1] > 0 else 0,
       abs(s[1]) if s[1] < 0 else 0,
       abs(s[2]) if s[2] > 0 else 0,
       abs(s[2]) if s[2] < 0 else 0,
       abs(s[3]) if s[3] > 0 else 0,
       abs(s[3]) if s[3] < 0 else 0 ])
#   print("Converting state: ", s, " ==> ", transformed)
  return transformed

#================================================

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

  # init variables
  state = env.reset()
  done = False
  score = 0
  reward = 0
  new_reward = 0
  step = 0
  # run episode, update online
  for _ in range(MAX_STEPS):
    nest.SetStatus(spike_recorder_ACTION_L, {"start":step*STEP, "stop":(step+1)*STEP})
    nest.SetStatus(spike_recorder_ACTION_R, {"start":step*STEP, "stop":(step+1)*STEP})
#     nest.SetStatus(spike_recorder_SNr_L, {"start":step*STEP, "stop":(step+1)*STEP})
#     nest.SetStatus(spike_recorder_SNr_R, {"start":step*STEP, "stop":(step+1)*STEP})


    # REWARD
    print("state: ", state)
    new_reward = max(10 * math.cos(20 * state[2]), 0)
    print("New reward : ", new_reward)
    amplitude_I_reward = new_reward
    print("Setting amplitude for reward: ", amplitude_I_reward, "     step: ", step)
    nest.SetStatus(dc_generator_reward, {"amplitude": amplitude_I_reward})

    # ENVIRONMENT
    new_transformed_state = transform_state(state)
    print("new_transformed_state", new_transformed_state)
    new_transformed_state_scaled = scaler.transform(new_transformed_state.reshape(1,-1)).reshape(-1)
    print("new_transformed_state_scaled:", new_transformed_state_scaled)
    dc_environment_current =  (np.exp(new_transformed_state_scaled)-1)*100.
    print("applying environment amplitude:", dc_environment_current)
    nest.SetStatus(dc_generator_env, {"amplitude": dc_environment_current})

    env.render()
    nest.Simulate(STEP)

    # REST for some time
    nest.SetStatus(dc_generator_env, {"amplitude": 0})
    step = step + 20
    nest.Simulate(20*STEP)


    left_spikes = len(spike_recorder_ACTION_L.get('events')['times'])
    right_spikes = len(spike_recorder_ACTION_R.get('events')['times'])
    print ("actor spikes2:", left_spikes, right_spikes, " at time ", step*STEP)

    action = 0 if left_spikes>right_spikes else 1

    print("Action:", action)


    new_state, reward, done, _ = env.step(action)


    # Hack reward
    #reward = 1- abs(new_state_scaled[2])
    if done:
        for i in range(0,1) :
          reward = 0
          step = step + 1
          nest.SetStatus(dc_generator_reward, {"amplitude": 0.})
          nest.Simulate(STEP)



    print("reward:", reward)

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

# np.savetxt('outputs/scores.txt', scores, delimiter=',')
# plt.figure(figsize=(5, 2.5), layout='constrained')
# nest.voltage_trace.from_device(voltmeter_STATE)
# plt.title("STATE")
# plt.show()
#
nest.raster_plot.from_device(spike_recorder_STATE, hist=True, title="STATE")
plt.show()
nest.raster_plot.from_device(spike_recorder_V, hist=True, title="V")
plt.show()
nest.raster_plot.from_device(spike_recorder_SNc, hist=True, title="SNc")
plt.show()
# nest.raster_plot.from_device(spike_recorder_SNr_L, hist=True, title="SNr_L")
# plt.show()
# nest.raster_plot.from_device(spike_recorder_SNr_R, hist=True, title="SNr_R")
# plt.show()
nest.raster_plot.from_device(spike_recorder_ACTION_L, hist=True, title="ACTION_L")
plt.show()
nest.raster_plot.from_device(spike_recorder_ACTION_R, hist=True, title="ACTION_R")
plt.show()

# plt.figure(figsize=(5, 2.5), layout='constrained')
# nest.voltage_trace.from_device(voltmeter_ACTION_L)
# plt.title("ACTION_L")
# plt.show()
# plt.figure(figsize=(5, 2.5), layout='constrained')
# nest.voltage_trace.from_device(voltmeter_ACTION_R)
# plt.title("ACTION_R")
# plt.show()

print("====== V === SNc ===")
print(nest.GetConnections(V, SNc))
print("====== STATE === V ===")
print(nest.GetConnections(STATE, V))
# print("====== SNr_L === ACTION_L ===")
# print(nest.GetConnections(SNr_L, ACTION_L))
# print("====== SNr_R === ACTION_R ===")
# print(nest.GetConnections(SNr_R, ACTION_R))

