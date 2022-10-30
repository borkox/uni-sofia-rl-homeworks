import nest
from nest import raster_plot as rplt
import numpy as np
import matplotlib.pyplot as plt

nest.ResetKernel()

num_neurons = 50
noise_weights = 2.1
ex_weights = 10.5
inh_weights = -2.6
ex_inh_weights = 2.8

action_left = nest.Create("iaf_psc_alpha", num_neurons)
action_right = nest.Create("iaf_psc_alpha", num_neurons)
wta_inhibitory = nest.Create("iaf_psc_alpha", num_neurons)

all_actor_neurons = action_left + action_right

n_input = nest.Create("poisson_generator", 10, {'rate': 3000.0})

nest.Connect(n_input, action_left, 'all_to_all', {'weight': noise_weights})
nest.Connect(n_input, action_right, 'all_to_all', {'weight': noise_weights})
nest.Connect(n_input, wta_inhibitory, 'all_to_all', {'weight': noise_weights * 0.9})
nest.Connect(action_left, action_left, 'all_to_all', {'weight': ex_weights})
nest.Connect(action_right, action_right, 'all_to_all', {'weight': ex_weights})

nest.Connect(all_actor_neurons, wta_inhibitory, 'all_to_all', {'weight': ex_inh_weights})
nest.Connect(wta_inhibitory, all_actor_neurons, 'all_to_all', {'weight': inh_weights})
sd = nest.Create("spike_recorder", 1)
spike_recorder_l = nest.Create("spike_recorder", 1)
spike_recorder_r = nest.Create("spike_recorder", 1)

nest.Connect(all_actor_neurons, sd, 'all_to_all')
nest.Connect(action_left, spike_recorder_l, 'all_to_all')
nest.Connect(action_right, spike_recorder_r, 'all_to_all')
# nest.Connect(wta_inhibitory, sd, 'all_to_all')
# nest.ResetNetwork()
seed = np.random.randint(0, 1000000)
nest.SetKernelStatus({'rng_seed': seed})
last_action_time = 0
nest.Simulate(500.0)
last_action_time += 500


for i in range(0,10):
  nest.SetStatus(n_input, {'rate': 0})
  nest.Simulate(30.0)
  last_action_time += 30
  nest.SetStatus(n_input, {'rate': 3000})
  nest.Simulate(500.0)
  rate_l = len([e for e in nest.GetStatus(spike_recorder_l, keys='events')[0]['times'] if e > last_action_time]) # calc the "firerate" of each actor population
  rate_r = len([e for e in nest.GetStatus(spike_recorder_r, keys='events')[0]['times'] if e > last_action_time]) # calc the "firerate" of each actor population
  print("left" if rate_l > rate_r else "right")
  last_action_time += 500

# rplt.from_device(sd, hist=False, title='WTA circuit')
nest.raster_plot.from_device(sd, hist=True, title="WTA circuit")
plt.show()
