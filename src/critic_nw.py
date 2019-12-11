import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNw(nn.Module):
    def __init__(self, name, beta, inp_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims, num_actions, chkp_dir='tmp/ddpg', action_bound=1, batch_size=64):
        super(CriticNw, self).__init__()
        self.lr = beta
        self.name = name
        self.inp_dims = inp_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.num_actions = num_actions
        self.action_bound = action_bound
        self.batch_size = batch_size
        self.chkp_file = os.path.join(chkp_dir,name+'_ddpg')

        # Input Layer
        self.fc1 = nn.Linear(self.inp_dims, self.fc1_dims)
        f1_bound = 1./np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1_bound, f1_bound)
        nn.init.uniform_(self.fc1.bias.data, -f1_bound, f1_bound)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        # Hidden Layer 1
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2_bound = 1./np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2_bound, f2_bound)
        nn.init.uniform_(self.fc2.bias.data, -f2_bound, f2_bound)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        f3_bound = 1./np.sqrt(self.fc3.weight.data.size()[0])
        nn.init.uniform_(self.fc3.weight.data, -f3_bound, f3_bound)
        nn.init.uniform_(self.fc3.bias.data, -f3_bound, f3_bound)
        self.bn3 = nn.LayerNorm(self.fc3_dims)

        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        f4_bound = 1./np.sqrt(self.fc4.weight.data.size()[0])
        nn.init.uniform_(self.fc4.weight.data, -f4_bound, f4_bound)
        nn.init.uniform_(self.fc4.bias.data, -f4_bound, f4_bound)
        self.bn4 = nn.LayerNorm(self.fc4_dims)

        self.action_value = nn.Linear(self.num_actions, fc4_dims)
        self.q = nn.Linear(self.fc4_dims, 1)
        f5_bound = 0.003
        nn.init.uniform_(self.q.weight.data, -f5_bound, f5_bound)
        nn.init.uniform_(self.q.bias.data, -f5_bound, f5_bound)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = F.relu(self.bn1(state_value))
        
        state_value = self.fc2(state_value)
        state_value = F.relu(self.bn2(state_value))

        state_value = self.fc3(state_value)
        state_value = F.relu(self.bn3(state_value))

        state_value = self.fc4(state_value)
        state_value = F.relu(self.bn4(state_value))

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint_to_file(self):
        print('############  Saving checkpoint  ##############')
        torch.save(self.state_dict(), self.chkp_file)

    def load_checkpoint_from_file(self):
        print('#############  Loading checkpoint  ###############')
        self.load_state_dict(torch.load(self.chkp_file))
