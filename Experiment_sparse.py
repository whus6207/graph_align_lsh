import matplotlib.pyplot as plt
import numpy as np 
from attr_utils import *
from io_utils import *
from lsh_utils import *
from io_sparse_utils import *
#from netalign_utils import *
import pandas as pd
import os.path
import pickle
import time

def experiment(df, filename = 'metadata/phys.edges', nodeAttributeFile = None,  
	multipleGraph = False, largeGraph = False, is_perm = False, 
	has_noise = False, noise_level = 0.05, plotAttribute = False, plotBucket = False, plotCorrectness = False, 
	GraphType = 'Directed', bandNumber = 2, adaptiveLSH = True, LSHType = 'Euclidean',
	loop_num = 3, cos_num_plane = 25, euc_width = 2, compute_hungarian = False, compute_sim = False, compute_netalign = False,
	threshold = 1):
	"""
	Experiment on two graphs with multiple setting

	"""
	start_preprocess = time.time()
	path = 'metadata/' + str(GraphType)

	A = loadSparseGraph(filename, GraphType)
	A, rest_idx = removeIsolatedSparse(A)


	if nodeAttributeFile is not None:
		nodeAttributesValue, nodeAttributesName = loadNodeFeature(nodeAttributeFile)
		#nodeAttributesValue = [nodeAttributesValue[i] for i in rest_idx]
	else:
		nodeAttributesValue, nodeAttributesName = [], []

	if multipleGraph == True:
		pass
	else:
		B, P = permuteSparse(A,is_perm, has_noise, noise_level)
	# Write edges to file
	writeSparseToFile(path+'/A.edges', A)
	writeSparseToFile(path+'/B.edges', B)

	if GraphType == 'Undirected':
		attributesA = getUndirAttribute(path + '/A.edges')
		attributesA = addNodeAttribute(attributesA, nodeAttributesName, nodeAttributesValue)

		with open(path + '/attributesA', 'w') as f:
			for index, row in attributesA.iterrows():
				f.write(str(attributesA.ix[index]))

		attributesB = getUndirAttribute(path + '/B.edges')
		attributesB = addNodeAttribute(attributesB, nodeAttributesName, nodeAttributesValue, P)

		with open(path + '/attributesB', 'w') as f:
			for index, row in attributesB.iterrows():
				f.write(str(attributesB.ix[index]))


		attributes = ['Degree', 'NodeBetweennessCentrality', 'PageRank', 
		'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']
		attributes += nodeAttributesName
	elif GraphType == 'Directed':
		attributesA = getDirAttribute(path +'/A.edges')
		attributesA = addNodeAttribute(attributesA, nodeAttributesName, nodeAttributesValue)

		with open(path+'/attributesA', 'w') as f:
			for index, row in attributesA.iterrows():
				f.write(str(attributesA.ix[index]))

		attributesB = getDirAttribute(path +'/B.edges')
		attributesB = addNodeAttribute(attributesB, nodeAttributesName, nodeAttributesValue, P)

		with open(path+'/attributesB', 'w') as f:
			for index, row in attributesB.iterrows():
				f.write(str(attributesB.ix[index]))

		attributes = ['Degree', 'InDegree', 'OutDegree', 'NodeBetweennessCentrality', 
					  'PageRank', 'HubsScore', 'AuthoritiesScore',
					  'EgonetDegree', 'EgonetInDegree', 'EgonetOutDegree',
					  'AvgNeighborDeg', 'AvgNeighborInDeg', 'AvgNeighborOutDeg','EgonetConnectivity']
		attributes += nodeAttributesName

	end_preprocess = time.time()
	preprocess_time = end_preprocess - start_preprocess
	
	print 'Pre-processing time: ' + preprocess_time

	return 

if __name__ == '__main__':
	adaptiveLSH = [False]
	noise = [False]
	bandNumber = [4]
	fname = 'exp_result_attr.pkl'
	df = pd.DataFrame(
			columns=['filename','nodeAttributeFile', 'is_perm', 'has_noise', 'GraphType'\
				, 'bandNumber', 'adaptiveLSH', 'LSHType', 'threshold'\
				, 'rank_score', 'rank_score_upper', 'correct_score', 'correct_score_upper', 'correct_score_hungarian'\
				, 'pairs_computed'])

	for a in adaptiveLSH:
		for n in noise:
			for b in bandNumber:
				experiment(df, filename = 'testdata/dblp.edges', nodeAttributeFile = None, 
					multipleGraph = False, is_perm = False, has_noise = n, plotAttribute = False, 
					plotBucket = False, plotCorrectness = False, GraphType = 'Undirected', bandNumber = b, 
					adaptiveLSH = a, LSHType = 'Cosine', compute_hungarian = False)
				# df = experiment(df, filename = 'testdata/dblp.edges', nodeAttributeFile = None, 
				# 	multipleGraph = False, is_perm = False, has_noise = n, plotAttribute = False, 
				# 	plotBucket = False, plotCorrectness = False, GraphType = 'Undirected', bandNumber = b, 
				# 	adaptiveLSH = a, LSHType = 'Euclidean', compute_hungarian = False)

	# pickle.dump(df, open(fname,'wb'))

	# writer = pd.ExcelWriter('exp_result_attr.xlsx')
	# df.to_excel(writer, sheet_name='Sheet1')
	# writer.save()
