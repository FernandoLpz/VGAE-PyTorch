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
   return tp/true

def accuracy(x_hat, test_edges, false_edges, node_to_idx):
   
   tp = 0
   for edge in test_edges:
      if x_hat[node_to_idx[edge[0]]][node_to_idx[edge[1]]] > 0.5:
         tp += 1
   
   f = 0   
   for false in false_edges:
      if x_hat[node_to_idx[false[0]]][node_to_idx[false[1]]] <0.5:
         f+=1
         
   acc = tp /len(test_edges)
   return acc

if __name__ == '__main__':
   
   graph_file = 'benchmarks/fb-pages-food/fb-pages-food_edges.csv'
	#graph_file = 'benchmarks/generic/generic.csv'
	
	# Load raw files and transform them into adjacency matrix
   pg = PrepareGraph(file=graph_file, test_size=0.1)
	
   # train_adj = torch.Tensor(pg.train_adj)
   # normalized = torch.Tensor(pg.normalized)
   # x_features = torch.Tensor(pg.x_features)

   # # Instatiate VGAE model
   # vgae = VGAE(num_neurons=16, num_features=pg.adjacency.shape[0], embedding_size=12)
   # optimizer = torch.optim.Adam(vgae.parameters(), lr=0.01)
	
   # for epoch in range(150):
	
   #    vgae.train()

   #    x_hat = vgae(train_adj, normalized, x_features)
   #    optimizer.zero_grad()

   #    loss = F.binary_cross_entropy(x_hat, train_adj)
   #    kl_divergence = 0.5/ x_hat.size(0) * (1 + 2*vgae.GCN_sigma - vgae.GCN_mu**2 - torch.exp(vgae.GCN_sigma)).sum(1).mean()
   #    loss -= kl_divergence

   #    loss.backward()
   #    optimizer.step()
      
   #    prec = precision(x_hat, train_adj)
   #    acc = accuracy(x_hat, pg.test_edges, pg.false_edges, pg.node_to_id)

   #    print('Epoch: ', epoch + 1, 'loss: ', loss.item(), 'precision: ', prec, 'acc: ', acc)