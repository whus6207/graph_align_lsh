from io_sparse_utils import *
import numpy.random as random
import pandas as pd
def add_node_att(inputfile, outputfile, number, graph_type = 'Undirected'):
	A = loadSparseGraph(inputfile, graph_type)
	A, rest_idx = removeIsolatedSparse(A)
	m, n = A.get_shape()
	col = ['a' + str(i) for i in range(number)]
	df = pd.DataFrame(columns = col)
	for i, c in enumerate(col):
		df[c] = [random.randint(i * 4 + 2) for _ in range(m)]
	df.to_csv(outputfile, index = False, header = True, sep = ' ')
add_node_att('Data/facebook.edges', 'test.nodes', 5)