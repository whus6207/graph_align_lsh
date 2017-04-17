import time
import os
import pandas as pd
import numpy as np
from utils.attr_utils import *
from utils.multi_sparse_utils import *
from scipy.sparse import identity
import scipy.sparse as sparse
import pickle
import sys

def preprocessing(edge_dir, node_dir = None, perm = False, save_dir = "", graph_type = 'Undirected',
	center_distance = 'canberra', findcenter = 0):
	path = './private_data/' + save_dir
	if not os.path.exists(path):
		os.makedirs(path)
	start_preprocess = time.time()
	real_path = 'metadata/realgraph/' + graph_type 
	multi_graphs = {}
	multi_perm = {}
	# Preprocess real graph
	i = 0
	for f in os.listdir(edge_dir):
		if not f.startswith('.'):
			A = loadSparseGraph(edge_dir + '/' + f, graph_type, weighted = True)
			# A, rest_idx = removeIsolatedSparse(A)
			if perm:
				p_graph, p = permuteMultiSparse(A, 1, graph_type, 0)
				multi_perm['M' + str(i)] = p[0]
				multi_graphs['M' + str(i)] = p_graph[0]
				A = p_graph[0]
			else:
				multi_graphs['M' + str(i)] = A
			writeSparseToFile(real_path + '/M'+ str(i) + '.edges', A)
			i += 1
	number = i - 1

	

			


	nodeAttributesValue, nodeAttributesName = [], []

	graph_attrs = {}

	if node_dir:
		nodeAttributesValue, nodeAttributesName = loadNodeFeature(node_dir)
		
	### get graph attributes
	if graph_type == 'Undirected':

		attributes = ['Degree', 'NodeBetweennessCentrality', 'PageRank', 
		'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']
		attributes += nodeAttributesName

		for key in multi_graphs.keys():
			node_num, n = multi_graphs[key].get_shape()
			attributesA = getUndirAttribute(real_path + '/' + key +'.edges', node_num)
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
	# save centers
	with open(path + '/centers', 'w') as f:
		for c in centers:
			f.write(c + '\n')
		f.close()

	with open(path + '/metadata', 'w') as f:
		f.write('graph_type' + " " + str(graph_type) + '\n')
		f.write('noise_level' + " " + str(0) + '\n')
		f.write('weighted_noise' + " " + str(0) + '\n')
		f.write('found_center' + " " + str(found_center) + '\n')
		f.write('number' + " " + str(number) + '\n')
		f.write('node_dir' + " " + str(node_dir) + '\n')
		f.write('center_distance' + " " + str(center_distance) + '\n')
		f.write('node_attribute_number' + " " + str(len(nodeAttributesName)) + '\n')
		f.close()

	pickle.dump(graph_attrs, open(path + '/attributes.pkl', 'wb'))
	pickle.dump(multi_graphs, open(path + '/multi_graphs.pkl', 'wb'))
	pickle.dump(multi_perm, open(path + '/permutations.pkl', 'wb'))


	end_preprocess = time.time()
	preprocess_time = end_preprocess - start_preprocess
	
	print 'Pre-processing time: ' + str(preprocess_time)

if __name__ == '__main__':
	if len(sys.argv) == 3:
		preprocessing(edge_dir = sys.argv[1], save_dir = sys.argv[2])
	elif len(sys.argv) == 4:
		preprocessing(edge_dir = sys.argv[1], save_dir = sys.argv[2], perm =True)
	elif len(sys.argv) == 5:
		preprocessing(edge_dir = sys.argv[1], node_dir = sys.argv[2], save_dir = sys.argv[3], perm = True)


	



