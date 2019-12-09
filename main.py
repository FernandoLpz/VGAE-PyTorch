import torch
import numpy as np
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from src import PrepareGraph
from src import VGAE


def accuracy(x_hat, pos_edges, neg_edges, node_to_idx):
	
	
	y_true = np.hstack((np.ones(len(pos_edges)), np.zeros(len(neg_edges))))
	y_pred = list()

	for edge in pos_edges:
		if x_hat[node_to_idx[edge[0]]][node_to_idx[edge[1]]] > 0.5:
			y_pred.append(1)
		else:
			y_pred.append(0)
	 
	for false in neg_edges:
		if x_hat[node_to_idx[false[0]]][node_to_idx[false[1]]] < 0.5:
			y_pred.append(0)
		else:
			y_pred.append(1)
			
	y_pred = np.array(y_pred)
	
	return accuracy_score(y_true, y_pred)

if __name__ == '__main__':
	
	graph_file = 'benchmarks/facebook_combined.csv'
	# graph_file = 'benchmarks/generic/generic.csv'
	pg = PrepareGraph(file=graph_file, test_size=0.1)
 
	train_adj = torch.Tensor(pg.train_adj)
	normalized = torch.Tensor(pg.normalized)
	x_features = torch.Tensor(pg.x_features)
	
	w1 = (train_adj.shape[0] * train_adj.shape[0] - train_adj.sum()) / train_adj.sum()
	w2 = train_adj.shape[0] * train_adj.shape[0] / (train_adj.shape[0] * train_adj.shape[0] - train_adj.sum())

	# Instatiate VGAE model
	vgae = VGAE(num_neurons=8, num_features=pg.adjacency.shape[0], embedding_size=4)
	optimizer = torch.optim.Adam(vgae.parameters(), lr=0.01)
	
	vgae.train()
	for epoch in range(200):

		optimizer.zero_grad()
		x_hat = vgae(train_adj, normalized, x_features)

		loss = w2 * F.binary_cross_entropy(x_hat, train_adj)
		kl_divergence = -(0.5/ x_hat.shape[0]) * (1 + 2*torch.log(vgae.GCN_sigma) - vgae.GCN_mu**2 - vgae.GCN_sigma**2).sum(1).mean()
		kl = F.kl_div(vgae.GCN_mu, vgae.GCN_sigma)
		loss -= kl_divergence

		loss.backward()
		optimizer.step()
		
		train_acc = accuracy(x_hat, pg.train_edges, pg.train_false_edges, pg.node_to_id)
		test_acc = accuracy(x_hat, pg.test_edges, pg.test_false_edges, pg.node_to_id)

		print('Epoch: ', epoch + 1, '\tloss: ', loss.item(), '\ttrain acc: ', train_acc, '\ttest acc: ', test_acc)