from lsh_utils import *
from io_sparse_utils import *
from attr_utils import *
import numpy.linalg
import snap
import pandas as pd
import numpy as np
import scipy.spatial.distance
from scipy import stats
from collections import defaultdict
import os

# A should be sparse matrix
def permuteMultiSparse(A, number, level):
	m, n = A.get_shape()
	multi_graph_w_permutation = []
	B = A.copy()
	noise = [(k, v) for k, v in zip(B.nonzero()[0], B.nonzero()[1]) if k <= v]
	visited = set(noise)
	scipy.random.shuffle(noise)
	noise = noise[: int(len(noise) * level) * number]
	# Dealing with existing edges
	multi_noise = [noise[len(noise) * i // number: len(noise) * (i+1) // number]for i in range(number)]
	for n in multi_noise:
		B = B.tolil()
		for i, j in n:
			B[i, j] = 0
			B[j, i] = 0
		# Adding edges
		for _ in range(2 * len(n)):  # Same amount as existing edges 
			add1, add2 = np.random.choice(m), np.random.choice(m)
			while ((add1, add2) in visited or (add2, add1) in visited):
				add1, add2 = np.random.choice(m), np.random.choice(m)
			B[add1, add2] = 1
			B[add2, add1] = 1
			visited.add((add1, add2))
		B = B.tocsr()
		multi_graph_w_permutation.append(B)
		B = A.copy()
	return multi_graph_w_permutation

def generate_multi_graph_synthetic(filename = None, graph_type = 'Undirected', number = 5, noise_level = 0.02):
	path = 'metadata/multigraph/'
	# multi_graph_w_permutation = []
	graph_info = {} # {graph name: adjacency matrix}
	if filename:
		A = loadSparseGraph(filename, graph_type)
	else:
		raise RuntimeError("Need an input file")
	A, rest_idx = removeIsolatedSparse(A)
	multi_graph_w_permutation = permuteMultiSparse(A, number, level = noise_level)
	writeSparseToFile(path + graph_type + '/M0.edges', A)
	graph_info['M0.edges'] = A
	for i, g in enumerate(multi_graph_w_permutation):
		writeSparseToFile(path + graph_type + '/M' + str(i+1) + '.edges', g)
		graph_info['M'+str(i+1)+'.edges'] = g

	return graph_info

def get_graph_signature(attributes):
	signature = []
	""" Extract features: Degree, EgonetDegree, Avg Egonet Neighbor, Egonet Connectivity, Clustering Coefficient  """
	for i in range(2, len(attributes.columns)): 
		# if i == 2 or i == 6:
		#  	continue
		feature = attributes.iloc[:, i]  
		# median
		md = np.median(feature)
		# mean
		mn = np.mean(feature)
		# std
		std_dev = np.std(feature)
		# skew
		skew = stats.skew(feature)
		# kurtosis
		krt = stats.kurtosis(feature)
		signature += [md, mn, std_dev, skew, krt]
		#signature += [md, mn, std_dev]
	return signature

def get_multi_graph_signature(graph_type = 'Undirected', graph_attrs = None):
	multigraph_sig = {}
	aggregations = {}
	if not graph_attrs:
		path = 'metadata/multigraph/'
		for filename in os.listdir(path + graph_type):
			if not filename.startswith('.'):
				aggregations[filename] = get_graph_feature(path, filename)
	else:
		aggregations = graph_attrs
	for graph, attr in aggregations.iteritems():
		multigraph_sig[graph] = get_graph_signature(attr)
	return multigraph_sig

def get_distribution_matrix(aggregations, attributes):
	m = len(aggregations)
	D = defaultdict(float)
	att = {}
	# attributes = aggregations[0].columns[2:]
	# attributes = ['Degree']
	for a in attributes:
		# for i in range(len(aggregations) - 1):
		# 	for j in range(i + 1, len(aggregations)):
		# 		D[i][j] = KL_sim(aggregations[i][a], aggregations[j][a])
		for g1, attr1 in aggregations.iteritems():
			for g2, attr2 in aggregations.iteritems():
				D[g1] += KL_sim(attr1[a], attr2[a])
		# D = D + D.T
		# att[a] = D
		# D = np.zeros((m, m))
	return D

def get_distance(sig1,sig2,type='canberra'):
	if type == 'canberra':
		return scipy.spatial.distance.canberra(sig1, sig2)
	elif type == 'manhattan':
		return numpy.linalg.norm(np.array(sig1) - np.array(sig2), ord=1)
	elif type == 'euclidean':
		return numpy.linalg.norm(np.array(sig1) - np.array(sig2))
	else:
		return cos_sim(sig1,sig2)

def get_distance_matrix_and_order(multigraph, check_center = True, distance = 'canberra'):
	# m = multigraph.keys()
	D = defaultdict(float)
	# if check_center:
	# 	m.remove('center.edges')
	# 	m = ['center.edges'] + m
	# for i, g1 in enumerate(m):
	# 	for j, g2 in enumerate(m):
	# 		if i <= j:
	# 			D[i][j] = get_distance(multigraph[g1], multigraph[g2], distance) 
	for g1, attr1 in multigraph.iteritems():
		for g2, attr2 in multigraph.iteritems():
			D[g1] += get_distance(attr1, attr2, distance)
	# D = D + D.T 
	return D
	
if __name__ == '__main__':
	GraphType = 'Undirected'
	path = 'metadata/multigraph/Undirected'
	multi_graphs = generate_multi_graph_synthetic('facebook/0.edges')
	graph_attrs = {}
	attributes = ['Degree', 'NodeBetweennessCentrality', 'PageRank', 
	'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']
	# attributes = ['Degree', 'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']
	node_num = multi_graphs['M0.edges'].get_shape()[0] # m of (m, n)
	for key in multi_graphs.keys():
		print key
		attributesA = getUndirAttribute(path + '/' + key, node_num)
		# TODO: handle when permutation possible
		with open(path + '/attributes'+key.split('.')[0], 'w') as f:
			for index, row in attributesA.iterrows():
				f.write(str(attributesA.ix[index]))
		graph_attrs[key] = attributesA[['Graph', 'Id']+attributes]
	#multigraph_sig = get_multi_graph_signature('Undirected', graph_attrs)
	#D = get_distance_matrix_and_order(multigraph_sig)
	graph_signatures = get_distribution_matrix(graph_attrs, attributes)
	print graph_signatures


