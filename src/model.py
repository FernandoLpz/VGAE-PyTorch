import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VGAE(nn.Module):
   def __init__(self, **kwargs):
      super(VGAE, self).__init__()
      self.num_neurons = kwargs['num_neurons']
      self.num_features = kwargs['num_features']
      self.embedding_size = kwargs['embedding_size']
      self.learning_rate = kwargs['learning_rate']
      
      self.w_0_mu = Variable(self.num_features, self.num_neurons)
      self.b_0_mu = Variable(self.num_neurons)
      self.w_1_mu = Variable(self.num_neurons, self.embedding_size)
      self.b_1_mu = Variable(self.embedding_size)
      
      self.w_0_sigma = Variable(self.num_features, self.num_neurons)
      self.b_0_sigma = Variable(self.num_neurons)
      self.w_1_sigma = Variable(self.num_neurons, self.embedding_size)
      self.b_1_sigma = Variable(self.embedding_size)
       
      
   def forward(self, adjacency, norm_adj):
      pass
   
   def encode(self):
      pass      
      