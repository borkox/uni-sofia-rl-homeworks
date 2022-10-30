import matplotlib.pyplot as plt
import environment as env

# we use environment class, number in argument is just an index for a certain environment we want to load
# 0 here is for the simple grid world environment with only three states and two actions to choose from
env.set_environment(0)

NUM_ITERATIONS = 10

nest.ResetKernel()
nest.set_verbosity("M_DEBUG")

seed = np.random.randint(0, 1000000)
nest.SetKernelStatus({'rng_seeds': range(seed, seed + 1), 'grng_seed': seed})

# Create states
world_dim = env.get_world_dimensions()
states = []
for i in range(world_dim['x']):
    states.append(nest.Create('iaf_psc_alpha', 100))


# Create actions
num_actions = env.get_num_possible_actions()
actions = []
for i in range(num_actions):
    actions.append(nest.Create('iaf_psc_alpha', 50))

all_actions = np.ravel(actions).tolist()


# Create WTA circuit
wta_ex_weights = 10.5
wta_inh_weights = -2.6
wta_ex_inh_weights = 2.8
wta_noise_weights = 2.1

wta_inh_neurons = nest.Create('iaf_psc_alpha', 50)

for i in range(len(actions)):
    nest.Connect(actions[i], actions[i], 'all_to_all', {'weight': wta_ex_weights})
    nest.Connect(actions[i], wta_inh_neurons, 'all_to_all', {'weight': wta_ex_inh_weights})

nest.Connect(wta_inh_neurons, all_actions, 'all_to_all', {'weight': wta_inh_weights})

wta_noise = nest.Create('poisson_generator', 10, {'rate': 3000.})
nest.Connect(wta_noise, all_actions, 'all_to_all', {'weight': wta_noise_weights})
nest.Connect(wta_noise, wta_inh_neurons, 'all_to_all', {'weight': wta_noise_weights * 0.9})

# Connect states to actions
nest.Connect(states[1], actions[0], 'all_to_all', {'weight': 0.0})
nest.Connect(states[1], actions[1], 'all_to_all', {'weight': 0.0})


# Create noise
noise = nest.Create('poisson_generator', 1, {'rate': 65000.})
nest.Connect(noise, np.ravel(states).tolist(), 'all_to_all', {'weight': 1.})


# Create state stimulus
stimulus = nest.Create('poisson_generator', 1, {'rate': 5000.})
position = env.get_agent_pos()['x']
nest.Connect(stimulus, states[position], 'all_to_all', {'weight': 1.})


# Create spike detector
sd_wta = nest.Create('spike_detector')
nest.Connect(all_actions, sd_wta)

sd_actions = nest.Create('spike_detector', num_actions)
for i in range(len(actions)):
    nest.Connect(actions[i], [sd_actions[i]], 'all_to_all')


# Main loop
actions_executed = 0
last_action_time = 0
while actions_executed < NUM_ITERATIONS:
    nest.SetStatus(wta_noise, {'rate': 3000.})
    nest.SetStatus(stimulus, {'rate': 5000.})
    nest.Simulate(900)
    max_rate = 0
    chosen_action = -1
    for i in range(len(sd_actions)):
        rate = len([e for e in nest.GetStatus([sd_actions[i]], keys='events')[0]['times'] if e > last_action_time]) # calc the "firerate" of each actor population
        if rate > max_rate:
            max_rate = rate # the population with the hightes rate wins
            chosen_action = i

    print ("iteration: ", actions_executed, ", action: ", chosen_action)
    nest.SetStatus(stimulus, {'rate': 0.})
    nest.SetStatus(wta_noise, {'rate': 0.})
    nest.Simulate(100.)
    last_action_time += 1000
    actions_executed += 1

rplt.from_device(sd_wta, title="WTA circuit as random walker")
rplt.show()