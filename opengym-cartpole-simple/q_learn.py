import numpy as np
import gym
from itertools import product
import math

"""
    ### Action Space
    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction of the fixed force the cart is pushed with.
    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |
    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it
    ### Observation Space
    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
    |-----|-----------------------|----------------------|--------------------|
    | Num | Observation           | Min                  | Max                |
    |-----|-----------------------|----------------------|--------------------|
    | 0   | Cart Position         | -4.8                 | 4.8                |
    | 1   | Cart Velocity         | -Inf                 | Inf                |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°)  | ~ 0.418 rad (24°)  |
    | 3   | Pole Angular Velocity | -Inf                 | Inf                |
    |-----|-----------------------|----------------------|--------------------|

"""

# Makes dictionary of all possible combinations of each vector component and the respective index
# {(0, 0, 0, 0): 0, (0, 0, 0, 1): 1 ... (2, 2, 5, 2): 161}
#ONE_HOT_COMBINATIONS = dict(zip(    product(range(3), range(5), range(11),range(5)), range(3*5*11*5)))
ONE_HOT_COMBINATIONS = dict(zip(    product(range(5), range(11),range(7)), range(5*11*7)))
X_DIM = len(ONE_HOT_COMBINATIONS) # 3*5*6*3 = 270

def encode(observation) :
    cart_position = observation[0]
    cart_velocity = observation[1]
    pole_angle    = math.degrees(observation[2])
    pole_angle_velocity = observation[3]

    x_pos = 0
    if cart_position <= -0.8 :
        x_pos = 0
    if -0.8 < cart_position and cart_position <= 0.8 :
        x_pos = 1
    if 0.8 < cart_position :
        x_pos = 2

    x_vel = 0
    if cart_velocity < -2 :
        x_vel = 0
    elif cart_velocity < -1 :
        x_vel = 1
    elif cart_velocity < 0:
        x_vel = 2
    elif cart_velocity < 1:
        x_vel = 3
    else:
        x_vel = 4

    th_angle = 0
    #print (pole_angle)
    if pole_angle < -6 :
        th_angle = 0
    elif pole_angle < 4:
        th_angle = 1
    elif pole_angle < 2:
        th_angle = 2
    elif pole_angle < 1:
        th_angle = 3
    elif pole_angle < 0.5:
        th_angle = 4
    elif pole_angle < 0:
        th_angle = 5
    elif pole_angle < 0.5:
        th_angle = 6
    elif pole_angle < 1:
        th_angle = 7
    elif pole_angle < 2:
        th_angle = 8
    elif pole_angle < 4:
        th_angle = 9
    else :
        th_angle = 10

    th_vel = 0
    if pole_angle_velocity < -2 :
        th_vel = 0
    if pole_angle_velocity < -1 :
        th_vel = 1
    elif pole_angle_velocity < -0.5 :
        th_vel = 2
    elif  pole_angle_velocity < 0:
        th_vel = 3
    elif  pole_angle_velocity < 0.5:
        th_vel = 4
    elif  pole_angle_velocity < 1:
        th_vel = 5
    else :
        th_vel = 6
    # Make one hot encoded
    arr = np.zeros(len(ONE_HOT_COMBINATIONS), dtype='int32')
    #print (f"x_pos={x_pos}, x_vel={x_vel}, th_angle={th_angle}, th_vel={th_vel}, x={ONE_HOT_COMBINATIONS[(x_pos,x_vel,th_angle,th_vel)]}")
    return ONE_HOT_COMBINATIONS[(x_vel,th_angle,th_vel)]


#
# Description of the environment:
#   https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
env = gym.make('CartPole-v1')
Q = np.zeros([X_DIM,env.action_space.n])
# 2. Parameters of Q-learning
eta = .628
gma = .9
epis = 5000
rev_list = [] # rewards per episode calculate
for i in range(epis):
    # Reset environment
    s = env.reset()
    s = encode(s)
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 200:
        env.render()
        j+=1
        # Choose action from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state & reward from environment
        observation,r,d,_ = env.step(a)
        s1 = encode(observation)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            print(f"Steps for episode: {j} last observation={observation}")
            break
    if j==200:
       print ("Win")
    rev_list.append(rAll)
    env.render()
print("Reward Sum on all episodes " + str(sum(rev_list)/epis))
print("Final Values Q-Table")
print(Q)

