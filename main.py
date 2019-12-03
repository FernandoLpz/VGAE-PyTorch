import torch

from src import PrepareGraph
from src import VGAE

if __name__ == '__main__':
   
   graph_file = 'benchmarks/fb-pages-food/fb-pages-food_edges.csv'
   
   # Load raw files and transform them into adjacency matrix
   pg = PrepareGraph(file=graph_file, test_size=0.1)
   
   train_adj = torch.FloatTensor(pg.train_adj)
   normalized = torch.FloatTensor(pg.normalized)
   x_features = torch.FloatTensor(pg.x_features)
   
   # Instatiate VGAE model
   vgae = VGAE(num_neurons=32,
            num_features=pg.train_adj.shape[0],
            embdding_size=32)
   
   optimizer = torch.optim.Adam(vgae.parameters(), lr=0.01)
   
   vgae.train()
   for epoch in range(10):
      x_reconstructed, sigma, mu = vgae(train_adj, normalized, x_features)
      optimizer.zero_grad()
      latent_loss = -(0.5/pg.train_adj.shape[0]) * tf.reduce_mean(tf.reduce_sum(1 + 2 * tf.log(sigma) - tf.square(mu) - tf.square(sigma), 1))           
      
   
   