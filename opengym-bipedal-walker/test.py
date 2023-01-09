
# Adapted from
# https://raw.githubusercontent.com/CiNCFZJ/CiNC/1a3060e33981b6154fa408ab904bff4ec0890fb2/summerschool2015/project/5%20Dopaminergic%20Error%20Signal%20With%20Critic.ipynb

# Dopaminergic Error Signal
# In this notebook, we will introduce a more plausible way to calculate the prediction error signal for reinforcement learning.
# We start with the same implementation as before but without calculating the values (and weights from states to actions) explicitly."


import nest
import nest.raster_plot as rplt
import numpy as np
import matplotlib.pyplot as plt
import environment as env
from mpl_toolkits.mplot3d import Axes3D

# env.set_environment(1)
# world_dim = env.get_world_dimensions()
# num_actions = env.get_num_possible_actions()
world_dim = {'x': 3, 'y': 5}
num_actions = 2

NUM_ITERATIONS = 300
NUM_STATE_NEURONS = 20
NUM_WTA_NEURONS = 50
WEIGHT_SCALING = 100 / NUM_STATE_NEURONS

nest.ResetKernel()
nest.set_verbosity("M_DEBUG")

rank = nest.Rank()
size = nest.NumProcesses() 
seed = np.random.randint(0, 1000000)
num_threads = 1
nest.SetKernelStatus({"local_num_threads": num_threads})
nest.SetKernelStatus({"rng_seeds": range(seed+num_threads * size + 1, seed + 2 * (num_threads * size) + 1),
                      "grng_seed": seed+size+num_threads,
                      "resolution": 0.1})
tau_pre = 20.
nest.SetDefaults("iaf_psc_alpha", {"tau_minus": tau_pre})

# Create states
states = []
for i in range(world_dim['x']):
    states.append([])
    for j in range(world_dim['y']):
        states[i].append(nest.Create('iaf_psc_alpha', NUM_STATE_NEURONS))
all_states = np.ravel(states).tolist()

# Create actions
actions = []
for i in range(num_actions):
    actions.append(nest.Create('iaf_psc_alpha', NUM_WTA_NEURONS))
all_actions = np.ravel(actions).tolist()

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
nest.CopyModel('neuromod_dopa_synapse', 'dopa_synapse', {'vt': vol_trans[0], 'A_LTP': 40., 'A_LTD': 50., "tau_plus": tau_post,
                                                         'Wmin': -10., 'Wmax': 10., 'b': 1., 'tau_n': tau_n, 'tau_c': tau_c})
        
nest.Connect(all_states, all_actions, 'all_to_all', {'model': 'dopa_synapse', 'weight': 0.0})

# TODO experimental: project from state to DA via critic 
nest.CopyModel('neuromod_dopa_synapse', 'dopa_synapse_critic', {'vt': vol_trans[0], 'A_LTP': 40., 'A_LTD': 50., "tau_plus": tau_post,
                                                                'Wmin': -10., 'Wmax': 10., 'b': 1., 'tau_n': tau_n, 'tau_c': tau_c})
critic = nest.Create('iaf_psc_alpha', 50)
nest.Connect(all_states, critic, 'all_to_all', {'model': 'dopa_synapse_critic', 'weight': 0.0})
nest.Connect(critic, DA_neurons, 'all_top_all', {'weight': -5., 'delay': 50.})
nest.Connect(critic, DA_neurons, 'all_to_all', {'weight': 5., 'delay': 1.})

critic_noise = nest.Create('poisson_generator', 1, {'rate': 65500.})
nest.Connect(critic_noise, critic)

# Create spike detector
sd_wta = nest.Create('spike_detector')
nest.Connect(all_actions, sd_wta)
nest.Connect(wta_inh_neurons, sd_wta)
sd_actions = nest.Create('spike_detector', num_actions)
for i in range(len(actions)):
    nest.Connect(actions[i], [sd_actions[i]])
sd_states = nest.Create('spike_detector')
nest.Connect(all_states, sd_states)
sd_DA = nest.Create('spike_detector', 1)
nest.Connect(DA_neurons, sd_DA, 'all_to_all')
sd_critic = nest.Create('spike_detector', 1)
nest.Connect(critic, sd_critic, 'all_to_all')


# Create noise
noise = nest.Create('poisson_generator', 1, {'rate': 65000.})
nest.Connect(noise, all_states, 'all_to_all', {'weight': 1.})
nest.Connect(noise, DA_neurons, 'all_to_all', {'weight': 1.0367})

def plot_values(fig, ax, position):
    plt.cla()
    
    values_plot = []      
    
    for i in range(world_dim['y']):
        values_plot.append([])
        for j in range(world_dim['x']):
            values_plot[i].append(np.mean([np.mean(nest.GetStatus(nest.GetConnections(states[j][i], actions[a]), 'weight')) for a in range(len(actions))]))
            if len(actions) == 4:
                q_north = np.mean(nest.GetStatus(nest.GetConnections(states[j][i], actions[0]), 'weight'))
                q_east = np.mean(nest.GetStatus(nest.GetConnections(states[j][i], actions[1]), 'weight'))
                q_south = np.mean(nest.GetStatus(nest.GetConnections(states[j][i], actions[2]), 'weight'))
                q_west = np.mean(nest.GetStatus(nest.GetConnections(states[j][i], actions[3]), 'weight'))
                ax.arrow(j, i, (q_east-q_west)/10., (q_south-q_north)/10., head_width=0.05, head_length=0.1, fc='k', ec='k')
            else:
                q_east = np.mean(nest.GetStatus(nest.GetConnections(states[j][i], actions[0]), 'weight'))
                q_west = np.mean(nest.GetStatus(nest.GetConnections(states[j][i], actions[1]), 'weight'))
                ax.arrow(j, i, (q_west-q_east)/10., 0., head_width=0.05, head_length=0.1, fc='k', ec='k')

    
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
    plt.draw()
    



# Main loop
actions_executed = 0
last_action_time = 0
in_end_position = False

# interactive plotting
fig, ax = plt.subplots()
plt.ion()
    
while actions_executed < NUM_ITERATIONS:
    position = env.get_agent_pos().copy()

    # plotting
    plot_values(fig, ax, position)
    
    if not in_end_position:
        nest.SetStatus(nest.GetConnections(stimulus, states[position['x']][position['y']]), {'weight': 1.})
        
        nest.SetStatus(wta_noise, {'rate': 3000.})

        nest.Simulate(400)
        max_rate = -1
        chosen_action = -1
        for i in range(len(sd_actions)):
            rate = len([e for e in nest.GetStatus([sd_actions[i]], keys='events')[0]['times'] if e > last_action_time]) # calc the \"firerate\" of each actor population
            if rate > max_rate:
                max_rate = rate # the population with the hightes rate wins
                chosen_action = i
        possible_actions = env.get_possible_actions() 

        new_position, outcome, in_end_position = env.move(possible_actions[chosen_action])

        print("iteration:", actions_executed, "action:", chosen_action)
        print("new pos:", new_position, "reward:", outcome)
        
        values = []
        for s in states:
            for ss in s:
                values.append(np.mean(nest.GetStatus(nest.GetConnections(ss, critic), 'weight')))

        print("values", values)
        
        # stimulate new state
        nest.SetStatus(nest.GetConnections(stimulus, states[position['x']][position['y']]), {'weight': 0.})

         
        # apply reward
        nest.SetStatus(nest.GetConnections(reward_stimulus, DA_neurons), {'weight': float(outcome)* WEIGHT_SCALING})
        nest.SetStatus(wta_noise, {'rate': 0.})
        
        # refactory time
        nest.Simulate(100.)
 
        nest.SetStatus(nest.GetConnections(stimulus, states[new_position['x']][new_position['y']]), {'weight': 1.})
        nest.SetStatus(nest.GetConnections(reward_stimulus, DA_neurons), {'weight': 0.0})
              
        last_action_time += 500
        actions_executed += 1
    else:      
        _, in_end_position = env.init_new_trial()
        nest.SetStatus(nest.GetConnections(stimulus, states[position['x']][position['y']]), {'weight': 0.})


rplt.from_device(sd_wta, title="WTA circuit")
rplt.from_device(sd_states, title="States")
rplt.from_device(sd_DA, title="DA pool")
rplt.from_device(sd_critic, title="Critic")
rplt.show()

