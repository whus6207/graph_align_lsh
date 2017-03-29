from io_utils import *
from lsh_utils import *
from attr_utils import getEgoAttr
import numpy.linalg
import snap
import pandas as pd
import numpy as np
import scipy.spatial.distance
from scipy import stats
import os

def permutemultiNoise(A, number, level):
	noise = np.zeros((len(A), len(A)))
	multi_graph_w_permutation = []
	noise_nodes = np.where(np.triu(np.random.choice([0, 1], size=(len(A), len(A)), p=[(100-level * number)/100, level * number /100])))
	noise_nodes = zip(noise_nodes[0], noise_nodes[1])
	np.random.shuffle(noise_nodes)
	multi_noise = [noise_nodes[len(noise_nodes) * i // number: len(noise_nodes) * (i+1) // number]for i in range(number)]
	for n in multi_noise:
		for i, j in n:
			noise[i][j] = 1
		B = (A + noise + noise.T) % 2
		multi_graph_w_permutation.append(B)
		noise = np.zeros((len(A), len(A)))
	return multi_graph_w_permutation

# A should be sparse matrix
def permuteSparse(A, number, level):
	m, n = A.get_shape()
	multi_graph_w_permutation = []
	B = A.copy()
	noise = [(k, v) for k, v in zip(B.nonzero()[0], B.nonzero()[1]) if k <= v]
	visited = set(noise)
	scipy.random.shuffle(noise) # [0] ????
	noise = noise[0][: int(len(noise[0]) * level) * number]
	# Dealing with existing edges
	multi_noise = [noise[len(noise) * i // number: len(noise) * (i+1) // number]for i in range(number)]
	for n in multi_noise:
		for i, j in n:
			B[i, j] = 0
			B[j, i] = 0
		# Adding edges
		for _ in range(int(m * m * level)):
			add1, add2 = np.random.choice(m), np.random.choice(m)
			while ((add1, add2) in visited):
				add1, add2 = np.random.choice(m), np.random.choice(m)
			B[add1, add2] = 1
			visited.add(add1, add2)
		multi_graph_w_permutation.append(B)
		B = A.copy()
	return multi_graph_w_permutation



def generate_multi_graph_synthetic(filename = None, graph_type = 'Undirected', number = 5):
	path = 'metadata/multigraph/'
	# multi_graph_w_permutation = []
	if filename:
		A = loadGraph(filename, graph_type)
	elif graph_type == 'Undirected':
		A = np.where(np.triu(np.random.rand(5,5), 1) >= 0.5, 1, 0)
		A += A.T
	else:
		A = np.where(np.triu(np.random.rand(5,5), 1) + np.tril(np.random.rand(5,5), -1) >= 0.5, 1, 0)
		#A = np.where(A - np.triu(A.T) >= 1, 1, 0) 
	A, rest_idx = removeIsolatedNodes(A)
	# for i in range(number):
	# 	multi_graph_w_permutation.append(permuteNoiseMat(A, is_perm = False, has_noise = True, level = 0.05))
	multi_graph_w_permutation = permutemultiNoise(A, number, level = 0.01)
	writeEdgesToFile(path + graph_type + '/center.edges', A)
	for i, g in enumerate(multi_graph_w_permutation):
		writeEdgesToFile(path + graph_type + '/M' + str(i) + '.edges', g)

	return A, multi_graph_w_permutation


def get_node_degree(UGraph, graph_type, attributes):
	degree = np.zeros((UGraph.GetNodes(),))
	OutDegV = snap.TIntPrV()
	snap.GetNodeOutDegV(UGraph, OutDegV)
	for item in OutDegV:
		degree[item.GetVal1()] = item.GetVal2()
	attributes['Degree'] = degree

def get_clustering_coeff(UGraph, attributes):
	coeff = np.zeros((UGraph.GetNodes(), ))
	for NI in UGraph.Nodes():
		i = NI.GetId()
		coeff[i] = snap.GetNodeClustCf(UGraph, i)
	attributes['ClusteringCoeff'] = coeff


def get_graph_feature(path, filename, graph_type = 'Undirected'):

	UGraph = snap.LoadEdgeList(snap.PUNGraph, path + graph_type + '/' + filename, 0, 1)
	attributes = pd.DataFrame(np.zeros(shape=(UGraph.GetNodes(), 7)),
		columns=['Graph', 'Id', 'Degree', 'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity', 'ClusteringCoeff'])
	attributes['Graph'] = [filename] * (UGraph.GetNodes())
	attributes['Id'] = range(1, UGraph.GetNodes()+1)
	# Get node degree
	get_node_degree(UGraph, graph_type, attributes)
	# Get 3 egonet features
	getEgoAttr(UGraph, attributes, directed = False)
	# Get clustering_coeff
	get_clustering_coeff(UGraph, attributes)

	return attributes	

def get_graph_signature(attributes):
	signature = []
	""" Extract features: Degree, EgonetDegree, Avg Egonet Neighbor, Egonet Connectivity, Clustering Coefficient  """
	for i in range(2, len(attributes.columns)): 
		if i == 2 or i == 6:
			continue
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

	
def get_multi_graph_signature(graph_type = 'Undirected'):
	multigraph = {}
	aggregations = []
	path = 'metadata/multigraph/'
	for filename in os.listdir(path + graph_type):
		if not filename.startswith('.'):
			aggregations.append(get_graph_feature(path, filename))
	for agg in aggregations:
		multigraph[agg['Graph'][0]] = get_graph_signature(agg)
	return multigraph

def get_canberra_distance(sig1,sig2):
	return scipy.spatial.distance.canberra(sig1, sig2)
	#return numpy.linalg.norm(np.array(sig1) - np.array(sig2))
	#return cos_sim(sig1,sig2)

def get_distance_matrix_and_order(multigraph, check_center = True):
	m = multigraph.keys()
	D = np.zeros((len(m), len(m)))
	if check_center:
		m.remove('center.edges')
		m = ['center.edges'] + m
	for i, g1 in enumerate(m):
		for j, g2 in enumerate(m):
			if i <= j:
				D[i][j] = get_canberra_distance(multigraph[g1], multigraph[g2]) 
	D = D + D.T 
	return D, m

def get_distribution_matrix(aggregations):
	m = len(aggregations)
	D = np.zeros((m, m))
	att = {}
	attributes = aggregations[0].columns[2:]
	attributes = ['Degree']
	for a in attributes:
		for i in range(len(aggregations) - 1):
			for j in range(i + 1, len(aggregations)):
				D[i][j] = KL_sim(aggregations[i][a], aggregations[j][a])
		D = D + D.T
		att[a] = D
		D = np.zeros((m, m))
	return att



def find_center(multigraph):
	"""
	rtype: string
	"""
	D, m = get_distance_matrix_and_order(multigraph)
	print(m)
	min_index = np.argmin(sum(D))
	return m[min_index]

if __name__ == '__main__':
	A, multigraph = generate_multi_graph_synthetic(filename = 'facebook/0.edges', graph_type = 'Undirected')
	multigraph = get_multi_graph_signature()
	print(sum(get_distance_matrix_and_order(multigraph)[0]))
	print(find_center(multigraph))
