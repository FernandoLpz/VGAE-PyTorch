import torch

class VGAE:
   def __init__(self, **kwargs):
      self.train_adj = kwargs['train_adj']
      self.normalized = kwargs['normalized']
      self.test_edges = kwargs['test_edges']
      self.false_edges = kwargs['false_edges']
      
      