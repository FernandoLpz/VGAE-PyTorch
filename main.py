import torch
import torch.nn.functional as F

from src import PrepareGraph
from src import VGAE

if __name__ == '__main__':
   
   graph_file = 'benchmarks/fb-pages-food/fb-pages-food_edges.csv'
   graph_file = 'benchmarks/generic/generic.csv'
   
   # Load raw files and transform them into adjacency matrix
   pg = PrepareGraph(file=graph_file, test_size=0.1)
   
   train_adj = torch.FloatTensor(pg.train_adj)
   normalized = torch.FloatTensor(pg.normalized)
   x_features = torch.FloatTensor(pg.x_features)
   
   # Instatiate VGAE model
   vgae = VGAE(num_neurons=2, num_features=pg.train_adj.shape[0], embedding_size=2)
   
   optimizer = torch.optim.Adam(vgae.parameters(), lr=0.001)
   
   for epoch in range(50):
      vgae.train()
      x_reconstructed = vgae(train_adj, normalized, x_features)
      
      loss = F.binary_cross_entropy(x_reconstructed, train_adj)
      kl_divergence = F.kl_div(vgae.GCN_mu, vgae.GCN_sigma)
      loss -= kl_divergence
      
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      
      print('Epoch: ', epoch, 'loss: ', loss.item())
   