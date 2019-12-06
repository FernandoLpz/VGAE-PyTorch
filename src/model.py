import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VGAE(nn.Module):
   def __init__(self, **kwargs):
      super(VGAE, self).__init__()
      
      self.num_neurons = kwargs['num_neurons']
      self.num_features = kwargs['num_features']
      self.embedding_size = kwargs['embedding_size']
      self.dropout = torch.nn.Dropout(0.5)

      #self.w_0 = torch.nn.Parameter(torch.randn(self.num_features, self.num_neurons, requires_grad=True))
      self.w_0 = VGAE.glorot_init(self.num_features, self.num_neurons)
      self.b_0 = torch.nn.Parameter(torch.randn(self.num_neurons, requires_grad=True))
      
      self.w_1_mu = VGAE.glorot_init(self.num_neurons, self.embedding_size)
      #self.w_1_mu = torch.nn.Parameter(torch.randn(self.num_neurons, self.embedding_size, requires_grad=True))
      self.b_1_mu = torch.nn.Parameter(torch.randn(self.embedding_size, requires_grad=True))

      self.w_1_sigma = VGAE.glorot_init(self.num_neurons, self.embedding_size)
      #self.w_1_sigma = torch.nn.Parameter(torch.randn(self.num_neurons, self.embedding_size, requires_grad=True))
      self.b_1_sigma = torch.nn.Parameter(torch.randn(self.embedding_size, requires_grad=True))
      
      # torch.nn.init.normal_(self.w_0)
      # torch.nn.init.normal_(self.w_1_mu)
      # torch.nn.init.normal_(self.w_1_sigma)
   
   @staticmethod 
   def glorot_init(input_dim, output_dim):
      
      init_range = np.sqrt(6.0/(input_dim + output_dim))
      initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
      
      return nn.Parameter(initial)

   def encode(self, adjacency, norm_adj, x_features):
      
      hidden_0 = torch.relu(torch.add(torch.matmul(torch.matmul(norm_adj, x_features), self.w_0), self.b_0))
      # hidden_0 = self.dropout(hidden_0)
      self.GCN_mu = torch.add(torch.matmul(torch.matmul(norm_adj, hidden_0), self.w_1_mu), self.b_1_mu)
      self.GCN_sigma = torch.exp(torch.add(torch.matmul(torch.matmul(norm_adj, hidden_0), self.w_1_sigma), self.b_1_sigma))

      z = self.GCN_mu + ( torch.randn(adjacency.size(0), self.embedding_size) * self.GCN_sigma )
      
      return z
   
   @staticmethod
   def decode(z):
      x_hat = torch.sigmoid(torch.matmul(z, z.t()))
      return x_hat


   def forward(self, adjacency, norm_adj, x_features):
      z = self.encode(adjacency, norm_adj, x_features)
      x_hat = VGAE.decode(z)
      
      return x_hat
      
           
      