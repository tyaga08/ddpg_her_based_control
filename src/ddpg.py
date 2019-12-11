import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import actor_nw
import critic_nw
import replay_buffer
import her
import noise

# need replay buffer class
# need a class for target Q network (function of s, a)
# we will use batch normalization
# policy would be deterministic, how to handle explore/exploit dilemna
# off policy learning done using stochastic policy
# replay buffer stores tuple of s_t, a_t, r_t, s_t+1. it must sample states at random to avoid sequence of subsequent steps
# There are 2 actor and 2 critic networks, a target for each
# soft updates, theta_prime = tau*theta_prime  + (1-tau)*theta
# target actor is the evaluation actor plus some noise. This is done for adding exploration. The noise is Ornstein Uhlenbeck. It models motion of particle in brownian motion class for noise
# The initial weights of the actor and critic network need to be constrained between a range that is mentioned in the paper 


""" CLASSES
    Replay Buffer
    Target Q Network
    OU Noise
    Actor
    Critic
"""

""" 
1. confusion between the actor and critic
2. When what is trained and backpropogated and why?
3. purpose of this before loss minimization: optimizer.zero_grad()
4. purpose of this: optimizer.step()
5. actor_obj_function = -self.critic.forward(states, mu_theta)............... why minus?
 """


class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, num_actions, layer1_size, layer2_size, layer3_size, layer4_size, output_dir, gamma=0.99, batch_size=64, max_size=100000):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.memory = replay_buffer.ReplayBuffer(max_size, input_dims, num_actions)
        print("max_size", max_size)
        self.her_memory = her.HERbuffer(max_size, input_dims, num_actions,env)
        
        self.actor = actor_nw.ActorNw('Actor', alpha, input_dims, layer1_size, layer2_size, layer3_size, layer4_size, num_actions, output_dir)

        self.critic = critic_nw.CriticNw('Critic', beta, input_dims, layer1_size, layer2_size, layer3_size, layer4_size, num_actions, output_dir)

        self.target_actor = actor_nw.ActorNw('TargetActor', alpha, input_dims, layer1_size, layer2_size, layer3_size, layer4_size, num_actions, output_dir)

        self.target_critic = critic_nw.CriticNw('TargetCritic', beta, input_dims, layer1_size, layer2_size, layer3_size, layer4_size, num_actions, output_dir)

        self.noise = noise.OUActionNoise(mu=np.zeros(num_actions))

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation = torch.tensor(observation,dtype=torch.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + torch.tensor(self.noise(),dtype=torch.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, state_dg, state_ag, new_state_dg, new_state_ag, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        self.her_memory.store_transition(state, action, reward, new_state, state_dg, state_ag, new_state_dg, new_state_ag, done)

    def initialize_her_buffer(self):
        self.her_memory.reset_counter()
    

    def learn(self):
        if self.memory.mem_cntr < self.batch_size*1:
            return 0, 0

        states, actions, new_states, rewards, done = self.memory.sample_buffer(self.batch_size)
        
        states = torch.tensor(states, dtype=torch.float).to(self.critic.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.critic.device)
        new_states = torch.tensor(new_states, dtype=torch.float).to(self.critic.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.critic.device)
        done = torch.tensor(done).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        # Q_phi(s,a)
        critic_value = self.critic.forward(states, actions)

        # mu_theta_targ(s')
        target_actions = self.target_actor.forward(new_states)

        # Q_phi_targ(s', mu_theta_targ(s'))
        critic_value_ = self.target_critic.forward(new_states, target_actions)

        # Find the target value for the batch size using the Bellman equation  
        target_value = []
        for idx in range(self.batch_size):
            target_value.append(rewards[idx] + self.gamma*done[idx]*critic_value_[idx])

        # print("tv_ddpg", target_value)

        target_value = torch.tensor(target_value, dtype=float).to(self.critic.device)
        target_value = target_value.view(self.batch_size,1)

        # Gradient descent to minimize the loss function
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target_value, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic.eval()

        # gradient ascent to find the optimal policy
        self.actor.optimizer.zero_grad()
        mu_theta = self.actor.forward(states)
        self.actor.train()
        actor_obj_function = -self.critic.forward(states, mu_theta)
        actor_obj_function = torch.mean(actor_obj_function)
        # print("actor_obj_function", actor_obj_function.item())
        actor_obj_function.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

        return critic_loss.item(), actor_obj_function.item()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        critic_state_dict = dict(critic_params)
        target_critic_state_dict = dict(target_critic_params)

        # Update step phi_targ = tau*phi_targ + (1 - tau)*phi 
        for key in critic_state_dict:
            critic_state_dict[key] = tau*critic_state_dict[key].clone() + (1-tau)*target_critic_state_dict[key].clone()
        self.target_critic.load_state_dict(critic_state_dict)

        # Update step theta_targ = tau*theta_targ + (1 - tau)*theta 
        for key in actor_state_dict:
            actor_state_dict[key] = tau*actor_state_dict[key].clone() + (1-tau)*target_actor_state_dict[key].clone()
        self.target_actor.load_state_dict(actor_state_dict)
    
    def save_models(self):
        self.actor.save_checkpoint_to_file()
        self.critic.save_checkpoint_to_file()
        self.target_actor.save_checkpoint_to_file()
        self.target_critic.save_checkpoint_to_file()

    def load_models(self):
        self.actor.load_checkpoint_from_file()
        self.critic.load_checkpoint_from_file()
        self.target_actor.load_checkpoint_from_file()
        self.target_critic.load_checkpoint_from_file()

    def check_actor_params(self):
        current_actor_params = self.actor.named_parameters()
        current_actor_dict = dict(current_actor_params)
        original_actor_dict = dict(self.original_actor.named_parameters())
        original_critic_dict = dict(self.original_critic.named_parameters())
        current_critic_params = self.critic.named_parameters()
        current_critic_dict = dict(current_critic_params)
        print('Checking Actor parameters')

        for param in current_actor_dict:
            print(param, T.equal(original_actor_dict[param], current_actor_dict[param]))
        print('Checking critic parameters')
        for param in current_critic_dict:
            print(param, T.equal(original_critic_dict[param], current_critic_dict[param]))
        input()

    