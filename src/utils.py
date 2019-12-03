import numpy as np
import pandas as pd
import networkx as nx

from random import randint

class PrepareGraph:
   def __init__(self, **kwargs):
      self.adjacency, self.edges = PrepareGraph.load_adjacency_matrix(kwargs['file'])
      self.train_adj, self.test_edges, self.false_edges = PrepareGraph.train_test_split(self.adjacency, kwargs['test_size'], self.edges)
      
   @staticmethod
   def train_test_split(adjacency, test_size, edges):
      
      test_edges = list()
      false_edges = list()
      
      train_adj = adjacency.copy()
      
      num_test_edges = round(len(edges) * test_size)
      
      # Create false edges
      while len(false_edges) < num_test_edges:
         row = randint(0, train_adj.shape[0] - 1)
         col = randint(0, train_adj.shape[0] - 1)
         
         if row != col:
            if tuple([row, col]) not in edges:
               if tuple([col, row]) not in edges:
                  if tuple([row, col]) not in false_edges:
                     if tuple([col, row]) not in false_edges:
                        false_edges.append(tuple([row, col]))
      
      # Create test edges
      idx_test = list()
      while len(idx_test) < num_test_edges:
         idx = randint(0, len(edges))
         if idx not in idx_test:
            idx_test.append(idx)
            
      for idx in idx_test:
         test_edges.append(edges[idx])
         adjacency[edges[idx][0], edges[idx][1]] = 0
         adjacency[edges[idx][1], edges[idx][0]] = 0
      
      return train_adj, test_edges, false_edges
      
   
   @staticmethod
   def load_adjacency_matrix(file):
 
      # Get the maximum number of entities
      def max_id(list_edges):
         maximum = 0
         for edge in list_edges:
            if edge[0] > maximum:
               maximum = edge[0]
            if edge[1] > maximum:
               maximum = edge[1]
         return maximum
      
      # Load csv file
      edges = pd.read_csv(file, header=None)
      edges = edges.values
      edges = [tuple(edge) for edge in edges]
      num_entities = max_id(edges)
      
      # Build adjacency matrix
      adjacency = np.zeros((num_entities + 1, num_entities + 1), dtype=int)
      for edge in edges:
         adjacency[edge[0], edge[1]] = 1
         adjacency[edge[1], edge[0]] = 1

      return adjacency, edges
