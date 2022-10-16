
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
DISCOUNT_FACTOR = 0.99
# number of episodes to run
NUM_EPISODES = 10
# max steps per episode
MAX_STEPS = 10000
# score agent needs for environment to be solved
SOLVED_SCORE = 195
# device to run model on

STEP = 7
scaler = scp.MinMaxScaler(feature_range=(0, 1), copy=True, clip=True)
# See https://www.gymlibrary.dev/environments/classic_control/cart_pole/#observation-space
scaler.fit([[-0.2,-1.5,-0.2,-3],[+0.2,+1.5,+0.2,+3]])

#================================================
nest.set_verbosity("M_WARNING")
nest.ResetKernel()
# ac_generator = nest.Create("ac_generator", params={ \
#     "origin": 0,
#     "frequency" : 150.0,
#     "offset": 100.,
#     "amplitude": 100.0
#     })

G_vt = nest.Create('volume_transmitter')

STATE = nest.Create("iaf_psc_alpha", 4, {"I_e": 10.0})
V      = nest.Create("iaf_psc_alpha", 40, {"I_e": 30.0})
G     = nest.Create("iaf_psc_alpha", 40, {"I_e": 30.0})
R     = nest.Create("iaf_psc_alpha", 10, {"I_e": 370.0})
ACTION_L  = nest.Create("iaf_psc_alpha", 10, {"I_e": 280.0})
ACTION_R  = nest.Create("iaf_psc_alpha", 10, {"I_e": 280.0})


dc_generator_reward = nest.Create('dc_generator', 1, {"amplitude": 20.})
dc_generator_env = nest.Create('dc_generator', 4, {"amplitude": 50.})
voltmeter_STATE = nest.Create("voltmeter")
voltmeter_V = nest.Create("voltmeter")
voltmeter_G = nest.Create("voltmeter")
spike_recorder_STATE = nest.Create('spike_recorder')
spike_recorder_V = nest.Create('spike_recorder')
spike_recorder_G = nest.Create('spike_recorder')
spike_recorder_R = nest.Create('spike_recorder')
spike_recorder_ACTION_L = nest.Create('spike_recorder')
spike_recorder_ACTION_R = nest.Create('spike_recorder')
nest.CopyModel('stdp_dopamine_synapse', 'dopsyn', \
               { 'vt': G_vt.get('global_id'), \
                'A_plus': 0.00001, 'A_minus': 0.00001, \
                'Wmin': .0, 'Wmax':2000.0, \
                'tau_c': STEP * 5, \
                'tau_n': STEP * 5, \
                'tau_plus': STEP * 3})

nest.Connect(dc_generator_env, STATE, 'one_to_one',
            syn_spec={'weight': 50 })

nest.Connect(STATE, V, \

            syn_spec={
            "weight": nest.random.uniform(min= 50., max=85.),
            'synapse_model': 'dopsyn'})

nest.Connect(STATE, G, \
            syn_spec={'weight': 30 })
# nest.Connect(G, G, \
#             conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.9},
#             syn_spec={'weight': nest.random.uniform(min= -20., max=-10.) })


# Reward signal
# nest.Connect(dc_generator_reward, G,'all_to_all', \
#              syn_spec={
#                    "weight": 50.})
nest.Connect(dc_generator_reward, R,'all_to_all', \
             syn_spec={
                   "weight": 180.})

# Volume transmitter
# nest.Connect(G, G_vt,'all_to_all')
# nest.Connect(V, G_vt,'all_to_all')
nest.Connect(R, G_vt,'all_to_all')



Gama=0.9
# Value function V(t)
nest.Connect(V, G, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
             syn_spec={'weight': -650.0, "delay": 1.0, 'synapse_model': 'dopsyn'})
# Value function V(t+1)
nest.Connect(V, G, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
             syn_spec={'weight': Gama*650.0, "delay": STEP + 1.0, 'synapse_model': 'dopsyn'})


# nest.Connect(V, ACTION_L, syn_spec={'weight':10.})
# nest.Connect(V, ACTION_R, syn_spec={'weight':10.})

nest.Connect(G, ACTION_L,
            conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.5},
            syn_spec={'weight': nest.random.uniform(min= 10., max=190.) })
nest.Connect(G, ACTION_R,
            conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.5},
            syn_spec={'weight': nest.random.uniform(min= 10., max=190.) })
# nest.Connect(G, G,
#             conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
#             syn_spec={'weight': nest.random.uniform(min= 10., max=20.) })



nest.Connect(voltmeter_STATE, STATE)
nest.Connect(voltmeter_V, V)
nest.Connect(voltmeter_G, G)

nest.Connect(STATE, spike_recorder_STATE)
nest.Connect(V, spike_recorder_V)
nest.Connect(G, spike_recorder_G)
nest.Connect(R, spike_recorder_R)
nest.Connect(ACTION_L, spike_recorder_ACTION_L)
nest.Connect(ACTION_R, spike_recorder_ACTION_R)
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
  step = 0
  # run episode, update online
  for _ in range(MAX_STEPS):
    step = step + 1
    nest.SetStatus(spike_recorder_ACTION_L, {"start":step*STEP, "stop":(step+1)*STEP})
    nest.SetStatus(spike_recorder_ACTION_R, {"start":step*STEP, "stop":(step+1)*STEP})
    print("Setting amplitude for reward: ", reward * 100)
#     nest.SetStatus(dc_generator_reward, {"start":step*STEP, "stop":(step+1)*STEP, "amplitude": min(score*0.8, 100.)})
    nest.SetStatus(dc_generator_reward, {"start":step*STEP, "stop":(step+1)*STEP, "amplitude": reward * 150})

    env.render()
    #time.sleep(0.03)
    # get action and log probability
    nest.Simulate(STEP)

#     action = random.randrange(2)
    left_spikes = len(spike_recorder_ACTION_L.get('events')['times'])
    right_spikes = len(spike_recorder_ACTION_R.get('events')['times'])
    print ("actor spikes:", left_spikes, right_spikes)

    action = 0 if left_spikes>right_spikes else 1
    print("Action:", action)

    # step with action
    #print(env.step(action))
    new_state, reward, done, _ = env.step(action)

    print("new_state:", new_state)
    new_state_scaled = scaler.transform(new_state.reshape(1,-1)).reshape(-1)
    print("new_state_scaled:", new_state_scaled)

    # Hack reward
    #reward = 1- abs(new_state_scaled[2])
    if done:
        for i in range(0,10) :
          reward = -1
          step = step + 1
          nest.SetStatus(dc_generator_reward, {"start":step*STEP, "stop":(step+1)*STEP, "amplitude": 0.})
          nest.Simulate(STEP)



    print("reward:", reward)

    print("applying ampltude:", (np.exp(new_state_scaled)-1)*20.)
    # Adjusting the generators to reflect environment
    nest.SetStatus(dc_generator_env, {"amplitude": (np.exp(new_state_scaled)-1)*200.})
    #print("Cortex[spikes]: ",spike_recorder_STATE.get('events')['times'][:-20])

    # update episode score
    score += reward

    # convert to torch tensor [Batch x observationsize]

    # get state value of next state

    # if terminal state, next state val is 0
    if done:
      print(f"Episode {episode} finished after {step} timesteps")


    # calculate policy loss

    if done:
      break

    # move into new state, discount I
    state = new_state

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
nest.raster_plot.from_device(spike_recorder_G, hist=True, title="G")
plt.show()
nest.raster_plot.from_device(spike_recorder_R, hist=True, title="R")
plt.show()

print("====== V === G ===")
print(nest.GetConnections(V, G))
print("====== STATE === V ===")
print(nest.GetConnections(STATE, V))
print("====== G === ACTION_L ===")
print(nest.GetConnections(G, ACTION_L))
print("====== G === ACTION_R ===")
print(nest.GetConnections(G, ACTION_R))

