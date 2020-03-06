import torch.nn as nn
import torch
from utils.math import *
from torch.distributions import Normal


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim,\
         hidden_size=(128, 128), log_std_min=-20, log_std_max=2,\
          activation='tanh', log_std=0, log_prob_head=0):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.log_prob_head = log_prob_head
        if self.log_prob_head == 1:
            init_w=3e-3
            self.action_log_std = nn.Linear(last_dim, action_dim)
            self.action_log_std.weight.data.uniform_(-init_w, init_w)
            self.action_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max


        self.epsilon = 1e-6

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        if self.log_prob_head == 1:
            action_log_std = self.action_log_std(x)
            action_log_std = torch.clamp(action_log_std, self.log_std_min, self.log_std_max)
        
        else:

            action_log_std = self.action_log_std.expand_as(action_mean)

        # print('ACTION_LOG_STD = ',action_log_std.mean())
        # print('ACTION_LOG_STD = ',action_log_std.std())

        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    
    
    
    def select_action_deterministic(self, x):
        action_mean, _, action_std = self.forward(x)
        # action = torch.tanh(action_mean).detach().cpu()#.numpy()

        return action_mean




    def select_action_stochastic(self, x):

        action_mean, _, action_std = self.forward(x)
         
        action = torch.normal(action_mean, action_std)

        # action = torch.tanh(action).detach().cpu()#.numpy()

        return action
     

    
   
    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)

        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    # def get_log_prob(self, x, actions):
    #     action_mean, action_log_std, action_std = self.forward(x)
    #     action_log_std = torch.clamp(action_log_std, self.log_std_min, self.log_std_max)
        # return normal_log_density(actions, action_mean, action_log_std, action_std), action_mean, action_std
    

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        # action_log_std = torch.clamp(action_log_std, self.log_std_min, self.log_std_max)
    
        normal = Normal(action_mean, action_std)
        z = normal.sample() ## Add reparam trick?
        action = torch.tanh(z)

        print('MEAN = ',action_mean.mean())
        print('STD = ',action_std.std())

        log_prob = normal.log_prob(z) #- torch.log(1 - action.pow(2) +  self.epsilon)
        # log_prob = normal_log_density(actions, action_mean, action_log_std, action_std).mean())
        log_prob = log_prob.sum(-1, keepdim=True)

        print('log_prob = ',log_prob.mean())
        print('normal log density = ',normal_log_density(actions, action_mean, action_log_std, action_std).mean())


        return log_prob,  action_mean, action_std

    def get_entropy(self,std):
        return normal_entropy(std)

    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}


