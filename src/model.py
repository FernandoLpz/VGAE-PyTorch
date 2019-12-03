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
      
      self.w_0_mu = Variable(self.num_features, self.num_neurons)
      self.b_0_mu = Variable(self.num_neurons)
      self.w_1_mu = Variable(self.num_neurons, self.embedding_size)
      self.b_1_mu = Variable(self.embedding_size)
      
      self.w_0_sigma = Variable(self.num_features, self.num_neurons)
      self.b_0_sigma = Variable(self.num_neurons)
      self.w_1_sigma = Variable(self.num_neurons, self.embedding_size)
      self.b_1_sigma = Variable(self.embedding_size)
      
      torch.nn.init.normal_(w_0_mu)
      torch.nn.init.normal_(w_1_mu)
      torch.nn.init.normal_(w_0_sigma)
      torch.nn.init.normal_(w_1_sigma)
      
      
   def encode(self, adjacency, norm_adj, x_features):
      hidden_0_mu = torch.relu(torch.add(torch.dot(torch.dot(norm_adj, x_features), self.w_0_mu), self.b_0_mu))
      self.mu = torch.add(torch.dot(torch.dot(norm_adj, hidden_0_mu), self.w_1_mu), self.b_1_mu)
      
      hidden_0_sigma = torch.relu(torch.add(torch.dot(torch.dot(norm_adj, x_features), self.w_0_sigma), self.b_0_sigma))
      self.sigma = torch.exp(torch.add(torch.dot(torch.dot(norm_adj, hidden_0_sigma), self.w_1_sigma), self.b_1_sigma))

      z = mu + np.random.normal(sigma.shape) * sigma
      
      return z
   
   @staticmethod
   def decode(z):
      x_reconstructed = torch.sigmoid(torch.dot(z, z.t()))
      return x_reconstructed


      
   def forward(self, adjacency, norm_adj, x_features):
      z = self.encode(adjacency, norm_adj, x_features)
      x_reconstructed = VGAE.decode(z)
      
      return x_reconstructed, self.sigma, self.mu
      
           
      