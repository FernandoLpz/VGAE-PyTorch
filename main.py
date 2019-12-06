import torch
import torch.nn.functional as F

from src import PrepareGraph
from src import VGAE


def precision(x_hat, x_true):
      tp, true = 0, 0
      for i in range(x_true.shape[0]):
            for j in range(x_true.shape[0]):
                  if x_true[i][j] == 1:
                        true += 1
                        if x_hat[i][j] > 0.5:
                              tp += 1
      return tp / true

if __name__ == '__main__':
   
   graph_file = 'benchmarks/fb-pages-food/fb-pages-food_edges.csv'
   # graph_file = 'benchmarks/generic/generic.csv'
   
   # Load raw files and transform them into adjacency matrix
   pg = PrepareGraph(file=graph_file, test_size=0.2)
   
   norm = pg.adjacency.shape[0] * pg.adjacency.shape[0] / float((pg.adjacency.shape[0] * pg.adjacency.shape[0] - pg.adjacency.sum()) * 2)
   
   train_adj = torch.FloatTensor(pg.train_adj)
   normalized = torch.FloatTensor(pg.normalized)
   x_features = torch.FloatTensor(pg.x_features)

   # Instatiate VGAE model
   vgae = VGAE(num_neurons=128, num_features=pg.adjacency.shape[0], embedding_size=128)
   optimizer = torch.optim.Adam(vgae.parameters(), lr=0.1)
   for epoch in range(50):
   
      vgae.train()
      
      x_hat = vgae(train_adj, normalized, x_features)
      loss = norm * F.binary_cross_entropy_with_logits(x_hat, train_adj)
      kl_divergence = F.kl_div(vgae.GCN_mu, vgae.GCN_sigma)
      loss -= kl_divergence
      
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      
      prec = precision(x_hat, pg.train_adj)
      print('Epoch: ', epoch + 1, 'loss: ', loss.item(), 'precision: ', prec)
      
      