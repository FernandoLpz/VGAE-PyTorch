from src import PrepareGraph
from src import VGAE

if __name__ == '__main__':
   
   graph_file = 'benchmarks/fb-pages-food/fb-pages-food_edges.csv'
   
   # Load raw files and transform them into adjacency matrix
   pg = PrepareGraph(file=graph_file, test_size=0.1)
   
   # Instatiate VGAE model
   vg = VGAE(train_adj=pg.train_adj,
             normalized=pg.normalized,
             test_edges=pg.test_edges,
             false_edges=pg.false_edges)