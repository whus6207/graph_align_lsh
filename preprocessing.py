import time
import os
import pandas as pd
import numpy as np
from attr_utils import *
from multi_sparse_utils import *
from scipy.sparse import identity
import h5py

def preprocessing(edge_dir, node_dir = None, save_dir = "", graph_type = 'Undirected'
	, number = 2, noise_level = 0.05, center_distance = 'canberra', findcenter = 0):
	path = './private_data/' + save_dir
	if not os.path.exists(path):
		os.makedirs(path)
	start_preprocess = time.time()

	multi_graphs, syn_path = generate_multi_graph_synthetic(filename = edge_dir, graph_type = graph_type, number = number, noise_level = noise_level)
	node_num, n = multi_graphs['M0.edges'].get_shape()
	nodeAttributesValue, nodeAttributesName = [], []
	P = identity(node_num)
	graph_attrs = {}

	if node_dir:
		nodeAttributesValue, nodeAttributesName = loadNodeFeature(node_dir)
		
	### get graph attributes
	if graph_type == 'Undirected':

		attributes = ['Degree', 'NodeBetweennessCentrality', 'PageRank', 
		'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']
		attributes += nodeAttributesName

		for key in multi_graphs.keys():
			attributesA = getUndirAttribute(syn_path + '/' + key, node_num)
			# TODO: handle when permutation possible
			attributesA = addNodeAttribute(attributesA, nodeAttributesName, nodeAttributesValue)
			attributesA = attributesA[['Graph', 'Id']+attributes]
			attributesA.to_hdf(path + '/attributes.h5', key.split('.')[0])

			graph_attrs[key] = attributesA

	elif graph_type == 'Directed':

		attributes = ['Degree', 'InDegree', 'OutDegree', 'NodeBetweennessCentrality', 
					  'PageRank', 'HubsScore', 'AuthoritiesScore',
					  'EgonetDegree', 'EgonetInDegree', 'EgonetOutDegree',
					  'AvgNeighborDeg', 'AvgNeighborInDeg', 'AvgNeighborOutDeg','EgonetConnectivity']
		attributes += nodeAttributesName

		for key in multi_graphs.keys():
			attributesA = getDirAttribute(psyn_pathath + '/' + key, node_num)
			attributesA = addNodeAttribute(attributesA, nodeAttributesName, nodeAttributesValue)
			attributesA = attributesA[['Graph', 'Id']+attributes]

			attributesA.to_hdf(path + '/attributes.h5', key.split('.')[0])

			graph_attrs[key] = attributesA



	graph_signatures = get_multi_graph_signature(graph_type, graph_attrs)
	centers = []
	found_center = find_center(graph_signatures, center_distance)
	print "found center: "+found_center
	if findcenter == 1:
		centers.append(found_center)
		if centers[0] != 'M0.edges':
			centers.append('M0.edges')
		else:
			print "found same center!!"
	elif findcenter == 0:
		centers = sorted(multi_graphs.keys())
	else:
		centers.append('M0.edges')
	print "check for center graph: {}".format(centers)

	# Save 
	# P = P.to_csr()
	# np.savez('Permutation.npz', data = P.data ,indices = P.indices, indptr = P.indptr, shape = P.shape )
	# save centers
	with open(path + '/centers', 'w') as f:
		for c in centers:
			f.write(c + '\n')

	end_preprocess = time.time()
	preprocess_time = end_preprocess - start_preprocess
	
	print 'Pre-processing time: ' + str(preprocess_time)

if __name__ == '__main__':
	preprocessing(edge_dir = 'Data/facebook.edges', save_dir = 'facebook')
	



