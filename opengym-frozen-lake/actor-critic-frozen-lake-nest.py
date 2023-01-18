import gym
from collections import deque
import nest.voltage_trace
import matplotlib.pyplot as plt
import numpy as np
from gym.envs.toy_text import FrozenLakeEnv


# number of episodes to run
NUM_EPISODES = 60
# max steps per episode
MAX_STEPS = 100
# Saves scores to file evey SAVE_SCORES_STEPS steps
SAVE_SCORES_STEPS = 5
# score agent needs for environment to be solved
SOLVED_MEAN_SCORE = 0.9
SOLVED_MIN_EPISODES = 3
# current time while it runs
current_time = 0
# STEP is milliseconds to run active simulation
STEP = 150
# REST_TIME is milliseconds to run rest for WTA and perform dopamine STDP
LEARN_TIME = 20
REST_TIME = 50
# Noise constants
NOISE_DA_NEURONS_WEIGHT = 0.01
NOISE_ALL_STATES_WEIGHT = 0.01
NOISE_RATE = 65000.
CRITIC_NOISE_RATE = 65500.
REWARD_STIMULUS_RATE = 65000.
STIMULUS_RATE = 65000.
WTA_NOISE_RATE = 500.

# ================================================
nest.set_verbosity("M_WARNING")
nest.ResetKernel()

# ================= Environment ==================
# Make environment
# env = gym.make('FrozenLake-v0')
env = FrozenLakeEnv(desc=["SFF",
                          "FFH",
                          "FFG"], is_slippery=False)
# env = FrozenLakeEnv(is_slippery=False)
WORLD_ROWS = env.nrow
WORLD_COLS = env.ncol
world_dim = {'x': WORLD_COLS, 'y': WORLD_ROWS}
num_actions = 4
possible_actions = [0, 1, 2, 3]
possible_actions_str = ["LEFT", "DOWN", "RIGHT", "UP"]
# ================================================

def plot_values(fig, ax, position):
    plt.cla()

    values_plot = []

    for i in range(world_dim['y']):
        values_plot.append([])
        for j in range(world_dim['x']):
            values_plot[i].append(np.mean([np.mean(nest.GetStatus(nest.GetConnections(states[j][i], actions[a]), 'weight')) for a in range(len(actions))]))
            if len(actions) == 4:
                q_left = np.mean(nest.GetStatus(nest.GetConnections(states[j][i], actions[0]), 'weight'))
                q_down = np.mean(nest.GetStatus(nest.GetConnections(states[j][i], actions[1]), 'weight'))
                q_right = np.mean(nest.GetStatus(nest.GetConnections(states[j][i], actions[2]), 'weight'))
                q_up = np.mean(nest.GetStatus(nest.GetConnections(states[j][i], actions[3]), 'weight'))
                ax.arrow(j, i, (q_right-q_left)/10., (q_down-q_up)/10., head_width=0.05, head_length=0.1, fc='k', ec='k')


    values_plot = np.array(values_plot)
    print(values_plot)

    plt.imshow(values_plot, interpolation='none', vmax=1 * WEIGHT_SCALING, vmin=-1 * WEIGHT_SCALING)

    xlabels = np.arange(0, len(states))
    ylabels = np.arange(0, len(states[0]))

    # Set the major ticks at the centers and minor tick at the edges
    xlocs = np.arange(len(xlabels))
    ylocs = np.arange(len(ylabels))
    ax.xaxis.set_ticks(xlocs + 0.5, minor=True)
    ax.xaxis.set(ticks=xlocs, ticklabels=xlabels)
    ax.yaxis.set_ticks(ylocs + 0.5, minor=True)
    ax.yaxis.set(ticks=ylocs, ticklabels=ylabels)

    # Turn on the grid for the minor ticks
    ax.grid(True, which='minor', linestyle='-', linewidth=2)

    for txt in ax.texts:
        txt.set_visible(False)

    ax.annotate(".", ((position['x'] + 0.5)/len(states), (1-(position['y'] + 0.5)/len(states[0]))), size=160, textcoords='axes fraction', color='white')
    #plt.draw()
# ================================================


NUM_STATE_NEURONS = 20
NUM_WTA_NEURONS = 50
# WEIGHT_SCALING = 100 / NUM_STATE_NEURONS
WEIGHT_SCALING = 100 * 20 / NUM_STATE_NEURONS

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
    for j in range(world_dim['y']):
        state_group = nest.Create('iaf_psc_alpha', NUM_STATE_NEURONS)
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

wta_noise = nest.Create('poisson_generator', 10, {'rate': WTA_NOISE_RATE})
nest.Connect(wta_noise, all_actions, 'all_to_all', {'weight': wta_noise_weights})
nest.Connect(wta_noise, wta_inh_neurons, 'all_to_all', {'weight': wta_noise_weights * 0.9})

# Create stimulus
stimulus = nest.Create('poisson_generator', 1, {'rate': STIMULUS_RATE})
nest.Connect(stimulus, all_states, 'all_to_all', {'weight': 0.})

# Here, we are implementing the dopaminergic nueron pool, volume transmitter and dopamin-modulated synapse between states and actions

# Create DA pool
DA_neurons = nest.Create('iaf_psc_alpha', 100)
vol_trans = nest.Create('volume_transmitter', 1, {'deliver_interval': 10})
nest.Connect(DA_neurons, vol_trans, 'all_to_all')

# Create reward stimulus
reward_stimulus = nest.Create('poisson_generator', 1, {'rate': REWARD_STIMULUS_RATE})
nest.Connect(reward_stimulus, DA_neurons, 'all_to_all', {'weight': 0.})

tau_c = 50.0
tau_n = 20.0
tau_plus = 20.

# Connect states to actions
nest.CopyModel('stdp_dopamine_synapse', 'dopa_synapse', {
    'vt': vol_trans.get('global_id'), 'A_plus': 1, 'A_minus': .5, "tau_plus": tau_plus,
    'Wmin': -10., 'Wmax': 10., 'b': 0., 'tau_n': tau_n, 'tau_c': tau_c})

nest.Connect(all_states, all_actions, 'all_to_all', {'synapse_model': 'dopa_synapse', 'weight': 0.0})

nest.CopyModel('stdp_dopamine_synapse', 'dopa_synapse_critic', {
    'vt': vol_trans.get('global_id'), 'A_plus': 1, 'A_minus': .5, "tau_plus": tau_plus,
    'Wmin': -10., 'Wmax': 10., 'b': 0., 'tau_n': tau_n, 'tau_c': tau_c})

critic = nest.Create('iaf_psc_alpha', 50)
nest.Connect(all_states, critic, 'all_to_all', {'synapse_model': 'dopa_synapse_critic', 'weight': 0.0})
nest.Connect(critic, DA_neurons, 'all_to_all', {'weight': -5., 'delay': 50})
nest.Connect(critic, DA_neurons, 'all_to_all', {'weight': 5., 'delay': 1.})

critic_noise = nest.Create('poisson_generator', 1, {'rate': CRITIC_NOISE_RATE})
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
noise = nest.Create('poisson_generator', 1, {'rate': NOISE_RATE})
nest.Connect(noise, all_states, 'all_to_all', {'weight': NOISE_ALL_STATES_WEIGHT})
nest.Connect(noise, DA_neurons, 'all_to_all', {'weight': NOISE_DA_NEURONS_WEIGHT})


# Init network
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space.n}")

# track scores
scores = []
# interactive plotting
fig, ax = plt.subplots()
# plt.ion()

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
    step = 0
    # run episode, update online
    for _ in range(MAX_STEPS):

        # ENVIRONMENT
        state_x = int(state % WORLD_COLS)
        state_y = int(state / WORLD_COLS)


        nest.SetStatus(nest.GetConnections(stimulus, states[state_x][state_y]), {'weight': 1.})
        nest.SetStatus(wta_noise, {'rate': 3000.})

        print("State position: ", state_x, ", ", state_y)
        # for si in range(len(states)):
        nest.SetStatus(nest.GetConnections(stimulus, states[state_x][state_y]), {'weight': 1.})

        env.render()
        nest.Simulate(STEP)

        max_rate = -1
        chosen_action = -1
        for i in range(len(sd_actions)):
            rate = len([e for e in nest.GetStatus(sd_actions[i], keys='events')[0]['times'] if
                        e > current_time])  # calc the \"firerate\" of each actor population
            if rate > max_rate:
                max_rate = rate  # the population with the hightes rate wins
                chosen_action = i

        current_time += STEP
        print("chose action:", possible_actions[chosen_action], " ", possible_actions_str[chosen_action], " at step ",
              step)

        action = possible_actions[chosen_action]
        new_state, reward, done, _ = env.step(action)

        # stimulate new state
        nest.SetStatus(nest.GetConnections(stimulus, states[state_x][state_y]), {'weight': 0.})

        # apply reward
        print("Scaled reward:", float(reward) * WEIGHT_SCALING)
        nest.SetStatus(nest.GetConnections(reward_stimulus, DA_neurons), {'weight': float(reward) * WEIGHT_SCALING})

        # learn time
        if reward > 0:
            print("Learn time")
            nest.Simulate(LEARN_TIME)
            current_time += LEARN_TIME

        nest.SetStatus(nest.GetConnections(reward_stimulus, DA_neurons), {'weight': 0.0})

        nest.SetStatus(wta_noise, {'rate': 0.})
        # refactory time
        nest.Simulate(REST_TIME)
        current_time += REST_TIME

        # plotting
        plot_values(fig, ax, {'x':state_x, 'y': state_y})


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
    if len(recent_scores) > SOLVED_MIN_EPISODES and np.array(recent_scores).mean() >= SOLVED_MEAN_SCORE:
        print("SOLVED")
        break
    else:
        print('Mean score: ', np.array(recent_scores).mean())
    if len(scores) % SAVE_SCORES_STEPS == 0:
        print("Save scores")
        np.savetxt('outputs/scores.txt', scores, delimiter=',')
    # if reward > 0:
    #     break
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
