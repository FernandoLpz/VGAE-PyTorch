import networkx as nx
import numpy as np
from random import randint

class PrepareGraph:
   def __init__(self, **kwargs):
      self.Graph = PrepareGraph.load_graph(kwargs['file'])
      self.adjacency = nx.adjacency_matrix(self.Graph)
      self.train, self.test = PrepareGraph.train_test_split(self.adjacency.copy(), kwargs['test_size'])
      
   @staticmethod
   def train_test_split(adjacency, test_size):
      false_edges = list()
      
      test_items = round(adjacency.shape[0] * test_size)
      
      idx_graph = np.argwhere(adjacency.toarray() == 1)
      idx_graph = [tuple(edge) for edge in idx_graph]
      
      while len(false_edges) < test_items:
         row = randint(0, adjacency.shape[0])
         col = randint(0, adjacency.shape[0])
         
         false_edge = tuple([row, col])
         
         if false_edge not in idx_graph:
            if false_edge not in false_edges:
               false_edges.append(false_edge)
      
      
      
      return 1,2
      
   
   @staticmethod
   def load_graph(file):
      Graph = nx.Graph()
      graph_file = open(file, 'r')

      return nx.parse_edgelist(graph_file, delimiter=',', create_using=Graph)
