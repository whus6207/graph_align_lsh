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
    degree = np.zeros((UGraph.GetMxNId(),))
    OutDegV = snap.TIntPrV()
    snap.GetNodeOutDegV(UGraph, OutDegV)
    for item in OutDegV:
        degree[item.GetVal1()] = item.GetVal2()
    attributes['Degree'] = degree

def get_clustering_coeff(UGraph, attributes):
	coeff = np.zeros((UGraph.GetMxNId(), ))
	for NI in UGraph.Nodes():
		i = NI.GetId()
		coeff[i] = snap.GetNodeClustCf(UGraph, i)
	attributes['ClusteringCoeff'] = coeff


def get_graph_feature(path, filename, graph_type = 'Undirected'):

	UGraph = snap.LoadEdgeList(snap.PUNGraph, path + graph_type + '/' + filename, 0, 1)
	attributes = pd.DataFrame(np.zeros(shape=(UGraph.GetMxNId(), 7)),
		columns=['Graph', 'Id', 'Degree', 'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity', 'ClusteringCoeff'])
	attributes['Graph'] = [filename] * (UGraph.GetMxNId())
	attributes['Id'] = range(1, UGraph.GetMxNId()+1)
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


def find_center(multigraph):
	"""
	rtype: string
	"""
	D, m = get_distance_matrix_and_order(multigraph)
	print(m)
	min_index = np.argmin(sum(D))
	return m[min_index]

if __name__ == '__main__':
	generate_multi_graph_synthetic(filename = 'facebook/0.edges', graph_type = 'Undirected')
	multigraph = get_multi_graph_signature()
	print(sum(get_distance_matrix_and_order(multigraph)[0]))
	print(find_center(multigraph))
