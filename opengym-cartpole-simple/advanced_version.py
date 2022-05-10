import numpy as np
import gym

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
X_DIM = 162 # 3*3*6*3
def on_hot_encode(observation) :
    cart_position = observation[0]
    cart_velocity = observation[1]
    pole_angle    = observation[2]
    pole_angle_velocity = observation[3]

    x_pos = 0
    if -2.4 < cart_position and cart_position <= -0.8 :
        x_pos = 0
    if -0.8 < cart_position and cart_position <= 0.8 :
        x_pos = 1
    if 0.8 < cart_position :
        x_pos = 2

    x_vel = 0
    if cart_velocity < -0.5 :
        x_vel = 1
    if -0.5 < cart_velocity and cart_velocity <= 0.5 :
        x_vel = 2
    if 0.5 < cart_velocity :
        x_vel = 3

    th_angle = 0
    if pole_angle < -6 :
        th_angle = 1
    if -6 < pole_angle and pole_angle <= -1 :
        th_angle = 2
    if -1 < pole_angle and pole_angle <= 0 :
        th_angle = 3
    if 0 < pole_angle and pole_angle <= 1 :
        th_angle = 4
    if 1 < pole_angle and pole_angle <= 6 :
        th_angle = 5
    if 6 < pole_angle :
        th_angle = 6

    th_vel = 0
    if pole_angle_velocity < -20 :
        th_vel = 1
    if -20 < pole_angle_velocity and pole_angle_velocity <= 20 :
        th_vel = 2
    if 20 < pole_angle_velocity :
        th_vel = 3
    # Make one hot encoded
    arr = np.zeros(X_DIM, dtype='int32')
    #print (f"x_pos={x_pos}, x_vel={x_vel}, th_angle={th_angle}, th_vel={th_vel}, x={x_pos*x_vel*th_angle*th_vel - 1}")
    arr[x_pos*x_vel*th_angle*th_vel - 1] = 1
    return arr


#exit()
#
# Description of the environment:
#   https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
env = gym.make('CartPole-v1')

E = np.zeros(X_DIM)
alpha = 0.0001
reward = 0
delta = 0.0001
gamma = 0.0001
beta = 0.0001
lamda = 0.0001
noise_denominator = 50
p_t_prev = 0

V = np.random.rand(X_DIM)/X_DIM
X_DASH = np.random.rand(X_DIM)/X_DIM
W = np.random.rand(X_DIM)/X_DIM
p_t = 0

for i_episode in range(800):
    observation = env.reset()
    for t in range(100):
        env.render()
        #print(observation)
        X = on_hot_encode(observation)
        p_t_prev = p_t
        p_t = np.dot(X, V)
        r_t_hat = reward + gamma*p_t - p_t_prev
        V = V + beta*r_t_hat*X_DASH
        X_DASH = lamda*X_DASH + (1-lamda)*X_DASH

        # 1 - right; 0 - left
        noise = (np.random.rand()-0.5)/noise_denominator
        y_t = np.dot(X_DASH, W)+noise
        f_z = 1 if y_t >= 0 else 0

        W = W + alpha*reward*E
        E = delta * E + (1-delta)*y_t*X

        #action = env.action_space.sample()
        action = np.array(f_z)
        #print (f" action: {action}")
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

