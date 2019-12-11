import numpy as np
import gym
import her

class ReplayBuffer(object):
    def __init__(self, max_size, inp_shape, num_actions):
        self.mem_size = max_size
        self.inp_shape = inp_shape
        self.num_actions = num_actions
        self.mem_cntr = 0
        self.state_mem = np.zeros((self.mem_size, inp_shape))
        self.new_state_mem = np.zeros((self.mem_size, inp_shape))
        self.action_mem = np.zeros((self.mem_size, self.num_actions))
        self.reward_mem = np.zeros(self.mem_size)
        self.terminal_mem = np.zeros(self.mem_size, dtype=float)
        self.buffer_flag = np.zeros(self.mem_size)
        self.max_size_reached = False

    def store_transition(self, state, action, reward, new_state, done=False, buffer_flag=0):
        index = self.mem_cntr % self.mem_size
        self.state_mem[index] = state
        self.action_mem[index] = action
        self.new_state_mem[index] = new_state
        self.reward_mem[index] = reward
        self.terminal_mem[index] = 1 - int(done)
        self.buffer_flag[index] = buffer_flag
        self.mem_cntr += 1
        if self.mem_cntr == self.mem_size:
            self.max_size_reached = True

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_mem[batch]
        new_states = self.new_state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        terminals = self.terminal_mem[batch]
        

        return states, actions, new_states, rewards, terminals

    def get_flag_ratio(self):
        ratio = 0.
        if not self.max_size_reached:
            for i in range(self.mem_cntr):
                ratio += self.buffer_flag[i]
            ratio = ratio/self.mem_cntr
        else:
            for i in range(self.mem_size):
                ratio += self.buffer_flag[i]
            ratio = ratio/self.mem_size
        
        return ratio
