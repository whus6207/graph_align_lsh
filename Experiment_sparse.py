import matplotlib.pyplot as plt
import numpy as np 
from attr_utils import *
from lsh_utils import *
from io_sparse_utils import *
from multi_sparse_utils import *
from scipy.sparse import identity
#from netalign_utils import *
import pandas as pd
import os.path
import pickle
import time

import warnings

def experiment(df, filename = 'Data/phys.edges', nodeAttributeFile = None,  
	multipleGraph = False, largeGraph = False, is_perm = False, 
	has_noise = False, noise_level = 0.05, 
	GraphType = 'Directed', bandNumber = 2, adaptiveLSH = True, LSHType = 'Euclidean',
	loop_num = 1, cos_num_plane = 25, euc_width = 2, compute_hungarian = False, compute_sim = False, compute_netalign = False,
	threshold = 1, center_distance = 'canberra',
	findcenter = 0): #findcenter = 1: find and check that one and original center; 0: check all, -1: original only
	"""
	Experiment on two graphs with multiple setting

	"""

	## debug
	np.seterr(all='raise')
	warnings.filterwarnings('error')

	start_preprocess = time.time()
	path = 'metadata/multigraph/' + str(GraphType)

	multi_graphs = generate_multi_graph_synthetic(filename = filename, graph_type = GraphType, number = 5, noise_level = noise_level)
	# graphs and the (structual and node) features of their nodes
	graph_attrs = {}
	# permutation does not make difference
	node_num, n = multi_graphs['M0.edges'].get_shape()
	P = identity(node_num)

	# get node attribute is file is specified
	if nodeAttributeFile is not None:
		nodeAttributesValue, nodeAttributesName = loadNodeFeature(nodeAttributeFile)
		#nodeAttributesValue = [nodeAttributesValue[i] for i in rest_idx]
	else:
		nodeAttributesValue, nodeAttributesName = [], []

	### get graph attributes
	if GraphType == 'Undirected':

		attributes = ['Degree', 'NodeBetweennessCentrality', 'PageRank', 
		'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']
		attributes += nodeAttributesName

		for key in multi_graphs.keys():
			attributesA = getUndirAttribute(path + '/' + key, node_num)
			# TODO: handle when permutation possible
			attributesA = addNodeAttribute(attributesA, nodeAttributesName, nodeAttributesValue)

			with open(path + '/attributes'+key.split('.')[0], 'w') as f:
				for index, row in attributesA.iterrows():
					f.write(str(attributesA.ix[index]))

			graph_attrs[key] = attributesA[['Graph', 'Id']+attributes]

	elif GraphType == 'Directed':

		attributes = ['Degree', 'InDegree', 'OutDegree', 'NodeBetweennessCentrality', 
					  'PageRank', 'HubsScore', 'AuthoritiesScore',
					  'EgonetDegree', 'EgonetInDegree', 'EgonetOutDegree',
					  'AvgNeighborDeg', 'AvgNeighborInDeg', 'AvgNeighborOutDeg','EgonetConnectivity']
		attributes += nodeAttributesName

		for key in multi_graphs.keys():
			attributesA = getDirAttribute(path + '/' + key, node_num)
			attributesA = addNodeAttribute(attributesA, nodeAttributesName, nodeAttributesValue)

			with open(path+'/attributesA', 'w') as f:
				for index, row in attributesA.iterrows():
					f.write(str(attributesA.ix[index]))

			graph_attrs[key] = attributesA[['Graph', 'Id']+attributes]



	graph_signatures = get_multi_graph_signature(GraphType, graph_attrs)
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

	end_preprocess = time.time()
	preprocess_time = end_preprocess - start_preprocess
	
	print 'Pre-processing time: ' + str(preprocess_time)
	

	for center_id in centers:
		rank_score = 0
		rank_score_upper = 0
		correct_score = 0
		correct_score_upper = 0
		correct_score_hungarian = 0
		pairs_computed = 0
		matching_time = 0
		avg_derived_rank = 0

		sim_matrix = {}
		if compute_sim:
			for g in graph_attrs.keys():
				sim_matrix[g] = computeWholeSimMat(graph_attrs[center_id], graph_attrs[g], LSHType)

		start_matching = time.time()
		# evaluate the accuracy and efficiency of our alg by generating buckets for <loop_num> times
		for i in range(loop_num):
			if GraphType == 'Undirected':
				if adaptiveLSH == True :
					bandDeg = ['Degree','PageRank','NodeBetweennessCentrality']
					bandEdge = ['EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']
					bandNode = nodeAttributesName[:]
						
					if LSHType == 'Cosine':
						bucketDeg = generateCosineBuckets(selectAndCombineMulti(graph_attrs, bandDeg), cos_num_plane)
						bucketEdge = generateCosineBuckets(selectAndCombineMulti(graph_attrs, bandEdge), cos_num_plane)

					elif LSHType == 'Euclidean':
						bucketDeg = generateEuclideanBuckets(selectAndCombineMulti(graph_attrs, bandDeg), euc_width)
						bucketEdge = generateEuclideanBuckets(selectAndCombineMulti(graph_attrs, bandEdge), euc_width)

					bucketNode = generateNodeBucketsMulti(LSHType, graph_attrs, bandNode, cos_num_plane, euc_width)
					buckets = [bucket for bucket in [bucketDeg, bucketEdge, bucketNode] if len(bucket) > 0]

					# for i, bucket in enumerate(buckets):
					# 	with open(path + '/buckets-band-' + str(i+1), 'w') as f:
					# 		for k, v in bucket.items():
					# 			f.write(str(k) + str(v) + '\n')
				else:
					band_all = list(attributes)
					np.random.shuffle(band_all)
					randomBand = [band_all[i*len(band_all)/bandNumber: (i + 1)*len(band_all)/bandNumber]for i in range(bandNumber)]

					buckets = []

					if LSHType == 'Cosine':
						for band in randomBand:
							buckets.append(generateCosineBuckets(selectAndCombineMulti(graph_attrs, band), cos_num_plane))


					elif LSHType == 'Euclidean':
						for band in randomBand:
							buckets.append(generateEuclideanBuckets(selectAndCombineMulti(graph_attrs, band), euc_width))

					# for i, bucket in enumerate(buckets):
					# 	with open(path + '/buckets-band-' + str(i+1), 'w') as f:
					# 		for k, v in bucket.items():
					# 			f.write(str(k) + str(v) + '\n')

			
			elif GraphType == 'Directed':
				if adaptiveLSH == True:
					bandDeg = ['Degree','InDegree','OutDegree']
					bandEgo = ['EgonetDegree', 'EgonetInDegree', 'EgonetOutDegree',
							  'AvgNeighborDeg', 'AvgNeighborInDeg', 'AvgNeighborOutDeg','EgonetConnectivity']
					bandCentr = ['PageRank', 'NodeBetweennessCentrality', 
								 'HubsScore', 'AuthoritiesScore']
					bandNode = nodeAttributesName[:]

					if LSHType == 'Cosine':

						bucketDeg = generateCosineBuckets(selectAndCombineMulti(graph_attrs, bandDeg), cos_num_plane)
						bucketEgo = generateCosineBuckets(selectAndCombineMulti(graph_attrs, bandEgo), cos_num_plane)
						bucketCentr = generateCosineBuckets(selectAndCombineMulti(graph_attrs, bandCentr), cos_num_plane)

					elif LSHType == 'Euclidean':
						bucketDeg = generateEuclideanBuckets(selectAndCombineMulti(graph_attrs, bandDeg), euc_width)
						bucketEgo = generateEuclideanBuckets(selectAndCombineMulti(graph_attrs, bandEgo), euc_width)
						bucketCentr = generateEuclideanBuckets(selectAndCombineMulti(graph_attrs, bandCentr), euc_width)

					bucketNode = generateNodeBuckets(LSHType, attributesA, attributesB, bandNode, cos_num_plane, euc_width)
					buckets = [bucket for bucket in [bucketDeg, bucketEgo, bucketCentr, bucketNode] if len(bucket) > 0]

					# for i, bucket in enumerate(buckets):
					# 	with open(path+'/buckets-band'+str(i+1), 'w') as f:
					# 		for k, v in bucket.items():
					# 			f.write(str(k) + str(v) + '\n')

				else:
					band_all = list(attributes)
					np.random.shuffle(band_all)
					randomBand = [band_all[i*len(band_all)/bandNumber: (i + 1)*len(band_all)/bandNumber]for i in range(bandNumber)]

					buckets = []

					if LSHType == 'Cosine':
						for band in randomBand:
							buckets.append(generateCosineBuckets(selectAndCombineMulti(graph_attrs, band), cos_num_plane))


					elif LSHType == 'Euclidean':
						for band in randomBand:
							buckets.append(generateEuclideanBuckets(selectAndCombineMulti(graph_attrs, band), euc_width))

					# for i, bucket in enumerate(buckets):
					# 	with open(path+'/buckets-band-'+str(i+1), 'w') as f:
					# 		for k, v in bucket.items():
					# 			f.write(str(k) + str(v) + '\n')


			stacked_attrs = selectAndCombineMulti(graph_attrs)	 
			pair_count_dict = combineBucketsBySumMulti(buckets, stacked_attrs[['Graph', 'Id']], graph_attrs.keys(), center_id)
			
			matching_matrix = {}
			this_pair_computed = {}
			Ranking = {}
			Best_Ranking = {}
			correctMatch = {}
			Best_correctMatch = {}
			hung_score = {}

			for g in pair_count_dict.keys():
				if g == center_id:
					continue
				matching_matrix[g], this_pair_computed[g]\
					= computeSparseMatchingMat(graph_attrs[center_id], graph_attrs[g], pair_count_dict[g], LSHType, threshold)
			

				Ranking[g], correctMatch[g] = sparseRank(matching_matrix[g], P)

				Best_Ranking[g] = Ranking[g]
				if compute_sim:
					Best_Ranking[g], Best_correctMatch[g] = sparseRank(sim_matrix[g], P)
				
				# Have to rewrite argmaxMatch!!!!!!!!!
				#correctMatch[g] = argmaxMatch(matching_matrix[g], graph_attrs[center_id], graph_attrs[g], P)
				Best_correctMatch[g] = correctMatch[g]
				# if compute_sim:
				# 	Best_correctMatch[g] = argmaxMatch(sim_matrix[g], graph_attrs[center_id], graph_attrs[g], P)
				hung_score[g] = correctMatch[g]
				if compute_hungarian:
					hung_score[g] = hungarianMatch(sim_matrix[g], P)


				rank_score += sum(Ranking[g])/len(Ranking[g])
				if compute_sim:
					rank_score_upper += sum(Best_Ranking[g])/len(Best_Ranking[g])
					correct_score_upper += sum(Best_correctMatch[g]) / float(len(Best_correctMatch[g]))
				else:
					rank_score_upper += 0
					correct_score_upper += 0
				correct_score += sum(correctMatch[g]) / float(len(correctMatch[g]))

				if compute_hungarian:
					correct_score_hungarian += sum(hung_score[g])/float(len(hung_score[g]))
				else:
					correct_score_hungarian += 0
				pairs_computed += this_pair_computed[g]/float(matching_matrix[g].shape[0]*matching_matrix[g].shape[1])

				print "=========================================================="
				print filename + ' ' + g + ', center:' + center_id + ', center_dist: '+center_distance
				print "has_noise = "+ str(has_noise)+", GraphType = "+ GraphType
				print "bandNumber = "+str(bandNumber)+", adaptiveLSH = "+ str(adaptiveLSH)+", LSHType = "+LSHType
				print "noise_level = "+str(noise_level)+", nodeAttributeFile = "+str(nodeAttributeFile)+", threshold = "+str(threshold)
				print "matching score by ranking: %f" %(sum(Ranking[g])/len(Ranking[g]))
				if compute_sim:
					print "matching score by ranking upper bound: %f" %(sum(Best_Ranking[g])/len(Best_Ranking[g]))
				print "matching score by correct match: %f" % (sum(correctMatch[g]) / float(len(correctMatch[g])))
				if compute_sim:
					print "matching score by correct match upper bound %f" % (sum(Best_correctMatch[g]) / float(len(Best_correctMatch[g])))
				if compute_hungarian:
					print "hungarian matching score upper bound: %f" %(sum(hung_score[g])/float(len(hung_score[g])))
				print "percentage of pairs computed: %f" %(this_pair_computed[g]/float(matching_matrix[g].shape[0]*matching_matrix[g].shape[1]))

			derived_matching_matrix = {}
			derived_rank = {}
			non_center = matching_matrix.keys()
			for i in xrange(len(non_center)):
				for j in xrange(i+1, len(non_center)):
					derived_matching_matrix[(non_center[i],non_center[j])] = matching_matrix[non_center[i]].T.dot(matching_matrix[non_center[j]])
					Ranking, correct_match = sparseRank(derived_matching_matrix[(non_center[i],non_center[j])], P, printing=False)
					derived_rank[(non_center[i],non_center[j])] = sum(Ranking)/len(Ranking)

			print 'derived rank score: '
			print derived_rank
			tmp_avg_derived_rank = sum([v for k,v in derived_rank.iteritems()])/len(derived_rank)
			avg_derived_rank += tmp_avg_derived_rank
			print 'avg derived rank score: ' + str(tmp_avg_derived_rank)

		rank_score /= loop_num * len(pair_count_dict.keys())
		rank_score_upper /= loop_num * len(pair_count_dict.keys())
		correct_score /= loop_num * len(pair_count_dict.keys())
		correct_score_upper /= loop_num * len(pair_count_dict.keys())
		correct_score_hungarian /= loop_num * len(pair_count_dict.keys())
		pairs_computed /= loop_num * len(pair_count_dict.keys())
		avg_derived_rank /= loop_num
		
		end_matching = time.time()
		matching_time = end_matching - start_matching		

		df = df.append({'filename':filename, 'nodeAttributeFile': str(nodeAttributeFile)\
			, 'has_noise':has_noise, 'noise_level':noise_level\
			, 'GraphType':GraphType, 'bandNumber':bandNumber, 'adaptiveLSH':adaptiveLSH, 'LSHType':LSHType\
			, 'threshold':threshold\
			, 'rank_score' : rank_score\
			, 'rank_score_upper' : rank_score_upper\
			, 'correct_score' : correct_score\
			, 'correct_score_upper' : correct_score_upper\
			, 'correct_score_hungarian' : correct_score_hungarian\
			, 'center_id': center_id\
			, 'found_center' : found_center\
			, 'avg_derived_rank': avg_derived_rank\
			, 'center_dist': center_distance\
			, 'pairs_computed' : pairs_computed\
			, 'preprocess_time': preprocess_time\
			, 'matching_time': matching_time\
			}, ignore_index=True)

	return df

if __name__ == '__main__':
	adaptiveLSH = [False]
	noise = [True]
	bandNumber = [2, 4, 8]
	LSH = ['Cosine', 'Euclidean']
	center_distance_types = ['canberra', 'manhattan', 'euclidean']
	fname = 'exp_result_multi_sparse.pkl'

	if os.path.isfile(fname):
		with open(fname, 'rb') as f:
			df = pickle.load(f)
	else:
		df = pd.DataFrame(
			columns=['filename','nodeAttributeFile', 'has_noise', 'GraphType'\
				, 'bandNumber', 'adaptiveLSH', 'LSHType', 'threshold'\
				, 'rank_score', 'rank_score_upper', 'correct_score', 'correct_score_upper', 'correct_score_hungarian'\
				, 'pairs_computed'])
	for dist_type in center_distance_types:
		df = experiment(df, filename = 'Data/facebook.edges', nodeAttributeFile = None, 
				has_noise = True, GraphType = 'Undirected', bandNumber = 2, 
				adaptiveLSH = False, LSHType = 'Cosine', noise_level = 0.001,
				center_distance = dist_type, findcenter = 0, threshold = 0.003)
		df = experiment(df, filename = 'Data/phys.edges', nodeAttributeFile = None, 
				has_noise = True, GraphType = 'Directed', bandNumber = 2, 
				adaptiveLSH = False, LSHType = 'Cosine', noise_level = 0.001,
				center_distance = dist_type, findcenter = 0, threshold = 0.003)
		# df = experiment(df, filename = 'Data/email.edges', nodeAttributeFile = None, 
		# 		has_noise = True, GraphType = 'Undirected', bandNumber = 2, 
		# 		adaptiveLSH = False, LSHType = 'Cosine', noise_level = 0.01,
		# 		center_distance = dist_type, findcenter = 0, threshold = 0.03)

	pickle.dump(df, open(fname,'wb'))

	writer = pd.ExcelWriter('exp_result_multi_sparse.xlsx')
	df.to_excel(writer, sheet_name='Sheet1')
	writer.save()