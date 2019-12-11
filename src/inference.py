import ddpg
import gym
import numpy as np
import matplotlib.pyplot as plt

def concat_obs_goal(state_dict):
    obs = state_dict['observation']
    goal = state_dict['desired_goal']
    _concat = np.concatenate([obs,goal])
    return _concat

# n_states = 28
# n_actions = 4

env = gym.make('FetchPush-v1')
print(dir(env))
observation = env.reset()

state_inp = concat_obs_goal(observation)

# parameters for the agent
d_alpha = 0.000025
d_beta = 0.00025
d_tau = 0.001
d_layer1_size = 256
d_layer2_size = 256
d_layer3_size = 256
d_layer4_size = 256
d_input_dims = state_inp.shape[0]
d_num_actions = env.action_space.shape[0]
d_output_dir = "tmp/fetch_test/"

fetch_agent = ddpg.Agent(d_alpha, d_beta, d_input_dims, d_tau, env, d_num_actions, d_layer1_size, d_layer2_size, d_layer3_size, d_layer4_size,d_output_dir)

fetch_agent.load_models()

score_history = []
num_steps = 50
test_no_episodes = 100

for i in range(test_no_episodes): #Total episodes to train
    
    observation = env.reset() #Sample desired goal

    desired_goal = observation["desired_goal"]
    curr_state = observation["observation"]
    # input_shape = obs["observation"].shape[0]
    # action_size = env.action_space.shape[0]
    # her = her_replay_buffer(num_steps, input_shape, action_size)

    for j in range(num_steps):
        curr_state_des_goal = concat_obs_goal(observation)
        act = fetch_agent.choose_action(curr_state_des_goal)
        new_state, reward, done, info = env.step(act)
        env.render()
        
        
    print("Episode number : {} Reward : {} Success : {}".format(i, reward,info['is_success']))

