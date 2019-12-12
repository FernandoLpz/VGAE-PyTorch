import os
import numpy as np
import pandas as pd
import networkx as nx

from random import randint

# Prepare data. Reads raw files (csv) and save it as list of edges
# Splits training and test sets as well as creates false edges
class PrepareGraph:
   def __init__(self, **kwargs):
      self.load_adjacency_matrix(kwargs['directory'], kwargs['dataset'])
      self.train_test_split(kwargs['test_size'])
      self.normalize_adjacency_matrix()
      self.x_features = np.identity(self.adjacency.shape[0])
      
   def normalize_adjacency_matrix(self):
      G = nx.from_numpy_matrix(self.train_adj)
      self.normalized = nx.normalized_laplacian_matrix(G)
      self.normalized = self.normalized.toarray()
      
      pass
      
   def train_test_split(self, test_size):
      
      self.train_edges = list()
      self.test_edges = list()
      self.test_false_edges = list()
      self.train_false_edges = list()
      
      self.train_adj = self.adjacency.copy()
      
      num_test_edges = round(len(self.edges) * test_size)
      
      # Create false edges
      while (len(self.train_false_edges) + len(self.test_false_edges)) < len(self.edges):
         row = randint(min(self.G.nodes()), max(self.G.nodes()))
         col = randint(min(self.G.nodes()), max(self.G.nodes()))
         
         if row != col:
            if tuple([row, col]) not in self.edges:
               if tuple([col, row]) not in self.edges:
                  if tuple([row, col]) not in self.train_false_edges:
                     if tuple([col, row]) not in self.train_false_edges:
                        if tuple([row, col]) not in self.test_false_edges:
                           if tuple([col, row]) not in self.test_false_edges:
                              if len(self.test_false_edges) < num_test_edges:
                                 self.test_false_edges.append(tuple([row, col]))
                              else:
                                 self.train_false_edges.append(tuple([row, col]))
                                 
      # Create test edges
      idx_test = list()
      while len(idx_test) < num_test_edges:
         idx = randint(0, len(self.edges)-1)
         if idx not in idx_test:
            idx_test.append(idx)

      for idx in idx_test:
         self.test_edges.append(self.edges[idx])
         self.train_adj[self.node_to_id[self.edges[idx][0]], self.node_to_id[self.edges[idx][1]]] = 0
         self.train_adj[self.node_to_id[self.edges[idx][1]], self.node_to_id[self.edges[idx][0]]] = 0
      
      # Train edges
      for i in range(self.train_adj.shape[0]):
         for j in range(self.train_adj.shape[1]):
            if self.train_adj[i][j] == 1:
               self.train_edges.append(tuple([self.id_to_node[i],self.id_to_node[j]]))
               
      pass
      
   def load_adjacency_matrix(self, directory, dataset):
      
      self.id_to_node = {}
      self.node_to_id = {}
      
      # Load csv file
      self.edges = pd.read_csv(os.path.join(directory, dataset, dataset+'.csv'), delimiter=' ',header=None)
      self.edges = self.edges.values
      self.edges = [tuple(edge) for edge in self.edges]
      
      self.G = nx.Graph()
      self.G.add_edges_from(self.edges)
      
      i=0
      for node in self.G.nodes():
         self.id_to_node[i] = node
         self.node_to_id[node] = i
         i+=1
         
      # Build adjacency matrix
      self.adjacency = np.zeros((len(self.G.nodes()), len(self.G.nodes())), dtype=int)
      for edge in self.edges:
         self.adjacency[self.node_to_id[edge[0]], self.node_to_id[edge[1]]] = 1
         self.adjacency[self.node_to_id[edge[1]], self.node_to_id[edge[0]]] = 1
      
      self.adjacency = self.adjacency + np.identity(len(self.G.nodes()))

      pass
