import numpy as np
import gym

class HERbuffer(object):
    def __init__(self, max_size, inp_shape, num_actions,env):
        self.mem_size = max_size
        self.inp_shape = inp_shape
        self.num_actions = num_actions
        self.mem_cntr = 0
        self.state_mem = np.zeros((self.mem_size, inp_shape))
        self.new_state_mem = np.zeros((self.mem_size, inp_shape))
        self.action_mem = np.zeros((self.mem_size, self.num_actions))
        self.reward_mem = np.zeros(self.mem_size)
        self.terminal_mem = np.zeros(self.mem_size, dtype=float)
        self.env = env
        self.state_dg = np.zeros((self.mem_size, 3))
        self.state_ag = np.zeros((self.mem_size, 3))
        self.new_state_dg = np.zeros((self.mem_size, 3))
        self.new_state_ag = np.zeros((self.mem_size, 3))
        

    def store_transition(self, state, action, reward, new_state, state_dg, state_ag, new_state_dg, new_state_ag,done=False):
        index = self.mem_cntr % self.mem_size
        self.state_mem[index] = state
        self.action_mem[index] = action
        self.new_state_mem[index] = new_state
        self.reward_mem[index] = reward
        self.terminal_mem[index] = 1 - int(done)
        self.state_dg[index] = state_dg
        self.state_ag[index] = state_ag
        self.new_state_dg[index] = new_state_dg
        self.new_state_ag[index] = new_state_ag
        self.mem_cntr += 1
        
    def manipulate_buffer(self):
        last_idx = self.mem_cntr
        new_desired_goal = self.new_state_ag[last_idx-1]

        # print("new_state_dg", new_desired_goal)
        for idx in range(self.mem_cntr):
            self.state_mem[idx][self.inp_shape-3:self.inp_shape] = new_desired_goal
            self.new_state_mem[idx][self.inp_shape-3:self.inp_shape] = new_desired_goal
            self.terminal_mem[idx] = 1
            self.reward_mem[idx] = -1
        self.terminal_mem[last_idx-4:last_idx] = 0
        self.reward_mem[last_idx-4:last_idx] = 0
        

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_mem[batch]
        new_states = self.new_state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        terminals = self.terminal_mem[batch]

        return states, actions, new_states, rewards, terminals

    def reset_counter(self):
        self.mem_cntr = 0

    def concat_obs_goal(self, state_dict):
        obs = state_dict['observation']
        goal = state_dict['desired_doal']
        _concat = np.concatenate([obs,goal])
        return _concat
    
    def return_elements(self):
        return self.state_mem[0:self.mem_cntr], self.action_mem[0:self.mem_cntr], self.new_state_mem[0:self.mem_cntr], self.reward_mem[0:self.mem_cntr], self.terminal_mem[0:self.mem_cntr],self.state_ag[0], self.new_state_ag[self.mem_cntr-1]