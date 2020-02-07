import ddpg
import os
import gym
import numpy as np
import matplotlib.pyplot as plt


_DIST_THRESHOLD = 0.0001

def concat_obs_goal(state_dict):
    obs = state_dict['observation']
    goal = state_dict['desired_goal']
    _concat = np.concatenate([obs,goal])
    return _concat

def are_posns_different(h_fag, h_lag):
    if abs(h_fag[0] - h_lag[0]) > _DIST_THRESHOLD:
        return True
    if abs(h_fag[1] - h_lag[1]) > _DIST_THRESHOLD:
        return True
    if abs(h_fag[2] - h_lag[2]) > _DIST_THRESHOLD:
        return True
    return False

# n_states = 25
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
d_output_dir = "tmp/Fetch_push_ddpg_her_2"

# Train time parameters
_MAX_EPOCHS = 2001
_MAX_EPISODES = 40
_MAX_TIMESTEPS = 50
print("d_input_dims", d_input_dims)

print("d_num_actions",d_num_actions)

# np.random.seed(0)

fetch_agent = ddpg.Agent(d_alpha, d_beta, d_input_dims, d_tau, env, d_num_actions, d_layer1_size, d_layer2_size, d_layer3_size, d_layer4_size,d_output_dir)

# fetch_agent.load_models()

score_history = []
critic_loss_history = []
actor_obj_history = []

score_file_name = "score.csv"
critic_loss_file_name = "critic_loss.csv"
actor_func_file_name = "actor_obj_func.csv"
time_count = 0

for _epochs in range(_MAX_EPOCHS):
    # first_state = env.reset()
    
    for _episodes in range(_MAX_EPISODES):
        observation = env.reset()
        # observation = first_state.copy()
        state_inp = concat_obs_goal(observation)
        done = False
        time_step = 0
        score = 0
        
        fetch_agent.initialize_her_buffer()

        while (not done):
            act = fetch_agent.choose_action(state_inp)
            new_state, reward, done, info = env.step(act)
            # env.render()
            fetch_agent.remember(concat_obs_goal(observation), act, reward, concat_obs_goal(new_state), observation['desired_goal'], observation['achieved_goal'], new_state['desired_goal'], new_state['achieved_goal'], int(done))
            score += reward
            observation = new_state
            time_step += 1
        score_history.append(score)
        
        fetch_agent.her_memory.manipulate_buffer()
        h_states, h_actions, h_new_states, h_rewards, h_terminal, h_fag, h_lag = fetch_agent.her_memory.return_elements()
        if are_posns_different(h_fag, h_lag):
            print("updating HER")
            for i in range(h_states.shape[0]):
                fetch_agent.memory.store_transition(h_states[i], h_actions[i], h_rewards[i], h_new_states[i], (h_rewards[i] == 0), 1)

    for _episodes in range(_MAX_EPISODES // 2):
        time_step = 0
        critic_loss = 0
        actor_objective_func = 0
        cl = 0.
        aof = 0.
        while (time_step < _MAX_TIMESTEPS):
            cl, aof = fetch_agent.learn()
            actor_objective_func += aof
            critic_loss += cl
            time_step += 1
        actor_objective_func = actor_objective_func/_MAX_TIMESTEPS
        critic_loss = critic_loss/_MAX_TIMESTEPS

        critic_loss_history.append(critic_loss)
        actor_obj_history.append(actor_objective_func    )

        print("//////////////////////////////////////////////////////")
        print('epoch', _epochs)
        print('episodes', _episodes)
        print('score %.2f' % score)
        print('trailing 100 games avg %.3f' %np.mean(score_history[-100:]))
        print('critic_loss', critic_loss)
        print('actor_objective_func', actor_objective_func)
        print('buffer_ratio', fetch_agent.memory.get_flag_ratio())
        print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")

    if  _epochs %5  == 0:
        fetch_agent.save_models()

        sc = open(os.path.join(d_output_dir, score_file_name),"w+")
        sc.write("Epochs: "+str(_epochs)+"\n")
        for i in range(len(score_history)):
            sc.write(str(score_history[i])+"\n")
        sc.close()

        cl = open(os.path.join(d_output_dir, critic_loss_file_name),"w+")
        cl.write("Epochs: "+str(_epochs)+"\n")
        for i in range(len(critic_loss_history)):
            cl.write(str(critic_loss_history[i])+"\n")
        cl.close()

        aoc = open(os.path.join(d_output_dir, actor_func_file_name),"w+")
        aoc.write("Epochs: "+str(_epochs)+"\n")
        for i in range(len(actor_obj_history)):
            aoc.write(str(actor_obj_history[i])+"\n")
        aoc.close()

print(score_history)
print(critic_loss_history)
print(actor_obj_history)


filename = 'fetch_slide.png'
plt.plot(score_history)
plt.xlabel("time_step")
plt.ylabel("score")
plt.savefig(os.path.join(d_output_dir, filename))


filename = 'fetch_slide_critic_loss.png'
plt.plot(critic_loss_history)
plt.xlabel("time_step")
plt.ylabel("critic_loss")
plt.savefig(os.path.join(d_output_dir, filename))


filename = 'fetch_slide_actor_obj_func.png'
plt.plot(actor_obj_history)
plt.xlabel("time_step")
plt.ylabel("actor_objective_function")
plt.savefig(os.path.join(d_output_dir, filename))