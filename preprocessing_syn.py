import time
import os
import pandas as pd
import numpy as np
from attr_utils import *
from multi_sparse_utils import *
from scipy.sparse import identity
import scipy.sparse as sparse
import pickle
import sys

def preprocessing(edge_dir, node_dir = None, save_dir = "", graph_type = 'Undirected',
	number = 5, noise_level = 0.01, weighted_noise = None, center_distance = 'canberra', findcenter = 0):
	path = './private_data/' + save_dir
	if not os.path.exists(path):
		os.makedirs(path)
	start_preprocess = time.time()

	multi_graphs, syn_path = generate_multi_graph_synthetic(filename = edge_dir, graph_type = graph_type, number = number, noise_level = noise_level, weighted_noise = weighted_noise)
	node_num, n = multi_graphs['M0'].get_shape()

	nodeAttributesValue, nodeAttributesName = [], []
	P = sparse.lil_matrix((node_num, n))
	for i in range(node_num):
		P[i, i] = 1
	P = P.tocsr()
	graph_attrs = {}

	if node_dir:
		nodeAttributesValue, nodeAttributesName = loadNodeFeature(node_dir)
		
	### get graph attributes
	if graph_type == 'Undirected':

		attributes = ['Degree', 'NodeBetweennessCentrality', 'PageRank', 
		'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']
		attributes += nodeAttributesName

		for key in multi_graphs.keys():
			attributesA = getUndirAttribute(syn_path + '/' + key +'.edges', node_num)
			# attributesA = getUndirAttribute(syn_path + '/' + key, node_num)
			# TODO: handle when permutation possible
			attributesA = addNodeAttribute(attributesA, nodeAttributesName, nodeAttributesValue)
			graph_attrs[key] = attributesA[['Graph', 'Id']+attributes]

	elif graph_type == 'Directed':

		attributes = ['Degree', 'InDegree', 'OutDegree', 'NodeBetweennessCentrality', 
					  'PageRank', 'HubsScore', 'AuthoritiesScore',
					  'EgonetDegree', 'EgonetInDegree', 'EgonetOutDegree',
					  'AvgNeighborDeg', 'AvgNeighborInDeg', 'AvgNeighborOutDeg','EgonetConnectivity']
		attributes += nodeAttributesName

		for key in multi_graphs.keys():
			attributesA = getDirAttribute(psyn_pathath + '/' + key +'.edges', node_num)
			# attributesA = getDirAttribute(psyn_pathath + '/' + key, node_num)
			attributesA = addNodeAttribute(attributesA, nodeAttributesName, nodeAttributesValue)
			graph_attrs[key] = attributesA[['Graph', 'Id']+attributes]

	with open(path + '/attributes', 'w') as f:
		for a in attributes:
			f.write(a + '\n')

	graph_signatures = get_multi_graph_signature(graph_type, graph_attrs)
	centers = []
	found_center = find_center(graph_signatures, center_distance)
	print "found center: "+found_center
	if findcenter == 1:
		centers.append(found_center)
		if centers[0] != 'M0':
			centers.append('M0')
		else:
			print "found same center!!"
	elif findcenter == 0:
		centers = sorted(multi_graphs.keys())
	else:
		centers.append('M0')

	if number == 1:
		centers = ['M0']
	print "check for center graph: {}".format(centers)

	# Save 
	# np.savez(path + '/Permutation.npz', data = P.data ,indices = P.indices, indptr = P.indptr, shape = P.shape )
	# save centers
	with open(path + '/centers', 'w') as f:
		for c in centers:
			f.write(c + '\n')
		f.close()
	# with open(path + '/found_center', 'w') as f:
	# 	f.write(found_center)
	# 	f.close()
	metadata = [graph_type, noise_level, weighted_noise, found_center, number]
	with open(path + '/metadata', 'w') as f:
		f.write('graph_type' + " " + str(graph_type) + '\n')
		f.write('noise_level' + " " + str(noise_level) + '\n')
		f.write('weighted_noise' + " " + str(weighted_noise) + '\n')
		f.write('found_center' + " " + str(found_center) + '\n')
		f.write('number' + " " + str(number) + '\n')
		f.write('node_dir' + " " + str(node_dir) + '\n')
		f.write('center_distance' + " " + str(center_distance) + '\n')
		f.close()

	pickle.dump(graph_attrs, open(path + '/attributes.pkl', 'wb'))


	end_preprocess = time.time()
	preprocess_time = end_preprocess - start_preprocess
	
	print 'Pre-processing time: ' + str(preprocess_time)

if __name__ == '__main__':
	preprocessing(edge_dir = sys.argv[1], number = int(sys.argv[3]), save_dir = sys.argv[2])

	


