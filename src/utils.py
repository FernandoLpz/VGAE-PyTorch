import networkx as nx

class PrepareGraph:
   def __init__(self, **kwargs):
      self.Graph = PrepareGraph.load_graph(kwargs['file'])
      self.adjacency = nx.adjacency_matrix(self.Graph)
      self.train, self.test = PrepareGraph.train_test_split(self.adjacency, kwargs['test_size'])
      
   @staticmethod
   def train_test_split(adjacency, test_size):
      print(type(adjacency))
      
   
   @staticmethod
   def load_graph(file):
      Graph = nx.Graph()
      graph_file = open(file, 'r')

      return nx.parse_edgelist(graph_file, delimiter=',', create_using=Graph)
