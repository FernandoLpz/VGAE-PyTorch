# Author: Fernando Lopez V

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Variational Graph Auto-Encoder
class VGAE(nn.Module):
   def __init__(self, **kwargs):
      super(VGAE, self).__init__()
      
      self.num_neurons = kwargs['num_neurons']
      self.num_features = kwargs['num_features']
      self.embedding_size = kwargs['embedding_size']
      
      self.w_0 = VGAE.random_uniform_init(self.num_features, self.num_neurons)
      self.b_0 = torch.nn.init.constant_(nn.Parameter(torch.Tensor(self.num_neurons)), 0.01)
      
      self.w_1_mu = VGAE.random_uniform_init(self.num_neurons, self.embedding_size)
      self.b_1_mu = torch.nn.init.constant_(nn.Parameter(torch.Tensor(self.embedding_size)), 0.01)

      self.w_1_sigma = VGAE.random_uniform_init(self.num_neurons, self.embedding_size)
      self.b_1_sigma = torch.nn.init.constant_(nn.Parameter(torch.Tensor(self.embedding_size)), 0.01)

      
   @staticmethod
   def random_uniform_init(input_dim, output_dim):
      
      init_range = np.sqrt(6.0/(input_dim + output_dim))
      tensor = torch.FloatTensor(input_dim, output_dim).uniform_(-init_range, init_range)
      
      return nn.Parameter(tensor)

   def encode(self, adjacency, norm_adj, x_features):
      
      hidden_0 = torch.relu(torch.add(torch.matmul(torch.matmul(norm_adj, x_features), self.w_0), self.b_0))
      self.GCN_mu = torch.add(torch.matmul(torch.matmul(norm_adj, hidden_0), self.w_1_mu), self.b_1_mu)
      self.GCN_sigma = torch.exp(torch.add(torch.matmul(torch.matmul(norm_adj, hidden_0), self.w_1_sigma), self.b_1_sigma))
      
      z = self.GCN_mu + torch.randn(self.GCN_sigma.size()) * self.GCN_sigma
      
      return z
   
   @staticmethod
   def decode(z):
      x_hat = torch.sigmoid(torch.matmul(z, z.t()))
      return x_hat


   def forward(self, adjacency, norm_adj, x_features):
      z = self.encode(adjacency, norm_adj, x_features)
      x_hat = VGAE.decode(z)
      
      return x_hat