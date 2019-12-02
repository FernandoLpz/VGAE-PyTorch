import networkx as nx
from random import randint

class PrepareGraph:
   def __init__(self, **kwargs):
      self.Graph = PrepareGraph.load_graph(kwargs['file'])
      self.adjacency = nx.adjacency_matrix(self.Graph)
      self.train, self.test = PrepareGraph.train_test_split(self.adjacency.copy(), kwargs['test_size'])
      
   @staticmethod
   def train_test_split(adjacency, test_size):
      test_items = round(adjacency.shape[0] * test_size)
      
      return 1,2
      
   
   @staticmethod
   def load_graph(file):
      Graph = nx.Graph()
      graph_file = open(file, 'r')

      return nx.parse_edgelist(graph_file, delimiter=',', create_using=Graph)
