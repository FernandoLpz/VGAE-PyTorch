from src import PrepareGraph

if __name__ == '__main__':
   
   graph_file = 'benchmarks/fb-pages-food/fb-pages-food_edges.csv'
   
   pg = PrepareGraph(file=graph_file, test_size=0.1)