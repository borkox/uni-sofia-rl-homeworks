
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
GAMA = 0.8
# number of episodes to run
NUM_EPISODES = 5
# max steps per episode
MAX_STEPS = 10000
# score agent needs for environment to be solved
SOLVED_SCORE = 195
# device to run model on

# STEP is milliseconds to simulate
STEP = 30

scaler = scp.MinMaxScaler(feature_range=(0, 1), copy=True, clip=True)
# See https://www.gymlibrary.dev/environments/classic_control/cart_pole/#observation-space
scaler.fit([[0,0,0,0,0,0,0,0],[+0.5,+0.5,+1.5,+1.5,+0.3,+0.3,+3,+3]])
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
nest.set_verbosity("M_WARNING")
nest.ResetKernel()


SNc_vt = nest.Create('volume_transmitter')

STATE = nest.Create("iaf_psc_alpha", 50, {"I_e": 10.0})
V      = nest.Create("iaf_psc_alpha", 50, {"I_e": 30.0})
SNc     = nest.Create("iaf_psc_alpha", 40, {"I_e": 150.0})
SNr_L     = nest.Create("iaf_psc_alpha", 10, {"I_e": 100.0})
SNr_R     = nest.Create("iaf_psc_alpha", 10, {"I_e": 100.0})
ACTION_L  = nest.Create("iaf_psc_alpha", 2, {"I_e": 700.0})
ACTION_R  = nest.Create("iaf_psc_alpha", 2, {"I_e": 700.0})


# NOISE  = nest.Create("dc_generator", 50, {"amplitude": 0.0})
dc_generator_reward = nest.Create('dc_generator', 1, {"amplitude": 20.})
dc_generator_env = nest.Create('dc_generator', 8, {"amplitude": 50.})
voltmeter_STATE = nest.Create("voltmeter")
voltmeter_V = nest.Create("voltmeter")
voltmeter_SNc = nest.Create("voltmeter")
voltmeter_ACTION_L = nest.Create("voltmeter")
voltmeter_ACTION_R = nest.Create("voltmeter")
spike_recorder_STATE = nest.Create('spike_recorder')
spike_recorder_V = nest.Create('spike_recorder')
spike_recorder_SNc = nest.Create('spike_recorder')
spike_recorder_SNr_L = nest.Create('spike_recorder')
spike_recorder_SNr_R = nest.Create('spike_recorder')
spike_recorder_ACTION_L = nest.Create('spike_recorder')
spike_recorder_ACTION_R = nest.Create('spike_recorder')
nest.CopyModel('stdp_dopamine_synapse', 'dopsyn', \
               { 'vt': SNc_vt.get('global_id'), \
                'A_plus': 0.01, 'A_minus': 0.01, \
                'Wmin': 0.0, 'Wmax':2000.0,'tau_c': 1000.0,
                                           'tau_n': 200.0,
                                           'tau_plus': 20.0})

nest.Connect(dc_generator_env, STATE,
            conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.5},
            syn_spec={'weight': 50 })

# nest.Connect(NOISE, V, \
#             conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
#             syn_spec={
#             "weight": nest.random.uniform(min= 10., max=20.)})
nest.Connect(STATE, V, \
            conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
            syn_spec={
            "weight": nest.random.uniform(min= 50., max=85.),
            'synapse_model': 'dopsyn'})
nest.Connect(STATE, V, \
            conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
            syn_spec={
            "weight": nest.random.uniform(min= -20., max=0.),
            'synapse_model': 'dopsyn'})

nest.Connect(dc_generator_reward, SNc,'all_to_all', \
             syn_spec={ "weight": 100.})

# Volume transmitter
nest.Connect(SNc, SNc_vt,'all_to_all')


# Value function V(t)
nest.Connect(V, SNc, 'all_to_all',
             syn_spec={'weight': -100.0, "delay": 1.0})
# Value function V(t+1)
nest.Connect(V, SNc, 'all_to_all',
             syn_spec={'weight': GAMA*100.0, "delay": STEP + 1.0})


nest.Connect(V, SNr_L,
            conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
            syn_spec={'weight': nest.random.uniform(min= 40., max=60.) })
nest.Connect(V, SNr_R,
            conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.8},
            syn_spec={'weight': nest.random.uniform(min= 40., max=60.) })



nest.Connect(SNr_L, ACTION_L,
            syn_spec={'weight': 150.0 })
nest.Connect(SNr_R, ACTION_R,
            syn_spec={'weight': 150.0 })
nest.Connect(SNr_L, SNr_R,
            syn_spec={'weight': -5.0 })
nest.Connect(SNr_R, SNr_L,
            syn_spec={'weight': -5.0 })

nest.Connect(ACTION_L, V,
            syn_spec={'weight': nest.random.uniform(min= 40., max=60.), 'synapse_model': 'dopsyn' })
nest.Connect(ACTION_R, V,
            syn_spec={'weight': nest.random.uniform(min= 40., max=60.), 'synapse_model': 'dopsyn' })


nest.Connect(voltmeter_STATE, STATE)
nest.Connect(voltmeter_V, V)
nest.Connect(voltmeter_SNc, SNc)
nest.Connect(voltmeter_ACTION_L, ACTION_L)
nest.Connect(voltmeter_ACTION_R, ACTION_R)

nest.Connect(STATE, spike_recorder_STATE)
nest.Connect(V, spike_recorder_V)
nest.Connect(SNc, spike_recorder_SNc)
nest.Connect(SNr_L, spike_recorder_SNr_L)
nest.Connect(SNr_R, spike_recorder_SNr_R)
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
    nest.SetStatus(spike_recorder_ACTION_L, {"start":step*STEP, "stop":(step+1)*STEP})
    nest.SetStatus(spike_recorder_ACTION_R, {"start":step*STEP, "stop":(step+1)*STEP})
    nest.SetStatus(spike_recorder_SNr_L, {"start":step*STEP, "stop":(step+1)*STEP})
    nest.SetStatus(spike_recorder_SNr_R, {"start":step*STEP, "stop":(step+1)*STEP})


    amplitude_I_reward = min(step * 0.01 * 50, 50 )
    print("Setting amplitude for reward: ", amplitude_I_reward, "     step: ", step)
    nest.SetStatus(dc_generator_reward, { "amplitude": amplitude_I_reward})
    #nest.SetStatus(NOISE, {"start":step*STEP, "stop":(step+1)*STEP, "amplitude": random.randint(-50,50)})

    env.render()
    #time.sleep(0.03)
    # get action and log probability
    nest.Simulate(STEP)

#     action = random.randrange(2)
#     left_spikes = len(spike_recorder_ACTION_L.get('events')['times'])
#     right_spikes = len(spike_recorder_ACTION_R.get('events')['times'])
#     print ("actor spikes:", left_spikes, right_spikes)
    left_spikes = len(spike_recorder_SNr_L.get('events')['times'])
    right_spikes = len(spike_recorder_SNr_R.get('events')['times'])
    print ("actor spikes2:", left_spikes, right_spikes, " at time ", step*STEP)

    action = 0 if left_spikes>right_spikes else 1
    print("Action:", action)

    # step with action
    #print(env.step(action))
    new_state, reward, done, _ = env.step(action)
    new_transformed_state = transform_state(new_state)

    print("new_state:", new_state)
    print("new_transformed_state", new_transformed_state)
    new_transformed_state_scaled = scaler.transform(new_transformed_state.reshape(1,-1)).reshape(-1)
    print("new_transformed_state_scaled:", new_transformed_state_scaled)

    # Hack reward
    #reward = 1- abs(new_state_scaled[2])
    if done:
        for i in range(0,10) :
          reward = -1
          step = step + 1
          nest.SetStatus(dc_generator_reward, { "amplitude": -100.})
#           nest.SetStatus(NOISE, {"start":step*STEP, "stop":(step+1)*STEP, "amplitude": random.randint(0,50)})
          nest.Simulate(STEP)



    print("reward:", reward)

    dc_environment_current =  (np.exp(new_transformed_state_scaled)-1)*100.
    print("applying environment amplitude:", dc_environment_current)
    # Adjusting the generators to reflect environment
    nest.SetStatus(dc_generator_env, {"amplitude": dc_environment_current})
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
nest.raster_plot.from_device(spike_recorder_SNr_L, hist=True, title="SNr_L")
plt.show()
nest.raster_plot.from_device(spike_recorder_SNr_R, hist=True, title="SNr_R")
plt.show()
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
print("====== SNr_L === ACTION_L ===")
print(nest.GetConnections(SNr_L, ACTION_L))
print("====== SNr_R === ACTION_R ===")
print(nest.GetConnections(SNr_R, ACTION_R))

