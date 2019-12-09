import torch
import numpy as np
import torch.nn.functional as F

from src import VGAE
from src import PrepareGraph
from sklearn.metrics import accuracy_score

def accuracy(x_pred, pos_edges, neg_edges, node_to_idx):
   y_true = np.hstack((np.ones(len(pos_edges)), np.zeros(len(neg_edges))))
   y_pred = list()
   
   for edge in pos_edges:
      if x_pred[node_to_idx[edge[0]]][node_to_idx[edge[1]]] > 0.5:
         y_pred.append(1)
      else:
         y_pred.append(0)
         
   for false in neg_edges:
      if x_pred[node_to_idx[false[0]]][node_to_idx[false[1]]] < 0.5:
         y_pred.append(0)
      else:
         y_pred.append(1)

   y_pred = np.array(y_pred)
   
   return accuracy_score(y_true, y_pred)

def load_data(args):
   return PrepareGraph(file=args.file, test_size=args.test_size)

def init_train(args, data):
   
   # Initialize tensors
   train_adj = torch.Tensor(data.train_adj)
   normalized = torch.Tensor(data.normalized)
   x_features = torch.Tensor(data.x_features)
   
   # Normalization
   w1 = (train_adj.shape[0] * train_adj.shape[0] - train_adj.sum()) / train_adj.sum()
   w2 = train_adj.shape[0] * train_adj.shape[0] / (train_adj.shape[0] * train_adj.shape[0] - train_adj.sum())
   
   # Init VGAE
   vgae = VGAE(num_neurons=args.num_neurons, num_features=data.adjacency.shape[0], embedding_size=args.embedding_size)
   
   # Define optimizer
   optimizer = torch.optim.Adam(vgae.parameters(), lr=0.01)
   
   # Init trainining phase
   vgae.train()
   
   for epoch in range(args.epochs):
      
      x_pred = vgae(train_adj, normalized, x_features)
      
      loss = w2 * F.binary_cross_entropy(x_pred, train_adj)
      kl_divergence = -(0.5/ x_pred.shape[0]) * (1 + 2*torch.log(vgae.GCN_sigma) - vgae.GCN_mu**2 - vgae.GCN_sigma**2).sum(1).mean()
      loss -= kl_divergence
      
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      
      train_acc = accuracy(x_pred, data.train_edges, data.train_false_edges, data.node_to_id)
      test_acc = accuracy(x_pred, data.test_edges, data.test_false_edges, data.node_to_id)
      
      print('Epoch: ', epoch + 1, '\tloss: ', loss.item(), '\ttrain acc: ', train_acc, '\ttest acc: ', test_acc)
      
   return

if __name__ == '__main__':
   
   data = load_data(args)
   init_train(args, data)