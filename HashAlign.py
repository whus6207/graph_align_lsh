import matplotlib.pyplot as plt
import numpy as np 
from attr_utils import *
from lsh_utils import *
from io_sparse_utils import *
from multi_sparse_utils import *
from scipy.sparse import identity
# from netalign_utils import *
import pandas as pd
#import h5py
import os.path
import pickle
import time

import warnings

sim_matrix = {}
Best_Ranking = {}
Best_correctMatch = {}

def experiment(df, filename, bandNumber = 2, adaptiveLSH = True, LSHType = 'Euclidean',
	loop_num = 1, cos_num_plane = 200, euc_width = 2, compute_hungarian = False, compute_sim = False, compute_netalign = False,
	threshold = 3.5): 
	"""
	Experiment on two graphs with multiple setting

	"""

	## debug
	np.seterr(all='raise')
	warnings.filterwarnings('error')

	# Load all necessary data
	metadata = {}
	centers = []
	found_center = None
	graph_attrs = {}
	graph_perm = {}
	# Load synthetic graph information
	with open('./private_data/' + filename + '/metadata') as f:
		for line in f:
			line = line.strip().split()
			metadata[line[0]] = line[1]

	# Check multiple graphs
	if metadata['number'] >= 1:	
		with open('./private_data/' + filename + '/centers') as f:
			for line in f:
				centers.append(line.strip().split()[0])
			f.close()
	else:
		raise RuntimeError("Need two graphs to align")

	# Load all graph attributes
	graph_attrs = pickle.load(open('./private_data/' + filename + '/attributes.pkl', 'rb'))
	graph_perm = pickle.load(open('./private_data/' + filename + '/permutations.pkl', 'rb'))
	multi_graphs = {}
	if compute_netalign:
		multi_graphs = pickle.load(open('./private_data/' + filename + '/multi_graphs.pkl', 'rb'))
	# Load attributes name
	attributes = []
	with open('./private_data/' + filename + '/attributes') as f:
		for line in f:
			attributes.append(line.strip().split()[0])
	# loader = np.load('./private_data/facebook/Permutation.npz')
	# P = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

	# Load permutation matrix 
	P = identity(len(graph_attrs['M0'])) # Should handle permutation case!!!!!!!!!!!

	for center_id in centers:
		rank_score = 0
		rank_score_upper = 0
		correct_score = 0
		correct_score_upper = 0
		correct_score_hungarian = 0
		netalign_score = 0
		pairs_computed = 0
		matching_time = 0
		avg_derived_rank = 0
		avg_derived_netalign = 0

		# sim_matrix = {}
		start_sim = time.time()
		if compute_sim:
			for g in graph_attrs.keys():
				if (center_id, g) not in sim_matrix and g != center_id:
					print '!!! computed sim_matrix !!!'
					sim_matrix[(center_id, g)] = computeWholeSimMat(graph_attrs[center_id], graph_attrs[g], LSHType)
				if (center_id, g) not in Best_Ranking and g != center_id:
                			Best_Ranking[(center_id, g)], Best_correctMatch[(center_id, g)] = sparseRank(sim_matrix[(center_id, g)], graph_perm[center_id], graph_perm[g])
		end_sim = time.time()
		print 'sim_time: '+str(end_sim-start_sim)
		start_matching = time.time()
		# evaluate the accuracy and efficiency of our alg by generating buckets for <loop_num> times
		for i in range(loop_num):
			if metadata['graph_type'] == 'Undirected':
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

			
			elif metadata['graph_type']  == 'Directed':
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
			correctMatch = {}
			hung_score = {}
			netalign_scores = {}

			for g in pair_count_dict.keys():
				if g == center_id:
					continue
				matching_matrix[g], this_pair_computed[g]\
					= computeSparseMatchingMat(graph_attrs[center_id], graph_attrs[g], pair_count_dict[g], LSHType, threshold)
			
				Ranking[g], correctMatch[g] = sparseRank(matching_matrix[g], graph_perm[center_id], graph_perm[g])
				
				if not compute_sim:
					Best_Ranking[(center_id, g)] = Ranking[g]
					Best_correctMatch[(center_id, g)] = correctMatch[g]
				
				# Have to rewrite argmaxMatch!!!!!!!!!
				#correctMatch[g] = argmaxMatch(matching_matrix[g], graph_attrs[center_id], graph_attrs[g], P)
				# if compute_sim:
				# 	Best_correctMatch[g] = argmaxMatch(sim_matrix[g], graph_attrs[center_id], graph_attrs[g], P)
				hung_score[g] = correctMatch[g]
				if compute_hungarian:
					hung_score[g] = hungarianMatch(sim_matrix[(center_id, g)], graph_perm[g])


				rank_score += sum(Ranking[g])/len(Ranking[g])
				if compute_sim:
					rank_score_upper += sum(Best_Ranking[(center_id, g)])/len(Best_Ranking[(center_id, g)])
					correct_score_upper += sum(Best_correctMatch[(center_id, g)]) / float(len(Best_correctMatch[(center_id, g)]))
				else:
					rank_score_upper += 0
					correct_score_upper += 0
				correct_score += sum(correctMatch[g]) / float(len(correctMatch[g]))

				if compute_hungarian:
					correct_score_hungarian += sum(hung_score[g])/float(len(hung_score[g]))
				else:
					correct_score_hungarian += 0
				pairs_computed += this_pair_computed[g]/float(matching_matrix[g].shape[0]*matching_matrix[g].shape[1])
				if compute_netalign:
					netalign_scores[g] = getNetalignScore(multi_graphs[center_id], multi_graphs[g], matching_matrix[g])
					netalign_score += netalign_scores[g]
				print "=========================================================="
				print filename + ' ' + g + ', center:' + center_id + ', center_dist: '+ metadata['center_distance']
				print "GraphType = " + metadata['graph_type'] 
				print "bandNumber = " + str(bandNumber) + ", adaptiveLSH = " + str(adaptiveLSH) + ", LSHType = " + LSHType
				print "noise_level = " + metadata['noise_level'] + ", nodeAttributeFile = " + metadata['node_dir'] + ", threshold = " + str(threshold)
				print "matching score by ranking: %f" %(sum(Ranking[g])/len(Ranking[g]))
				if compute_sim:
					print "matching score by ranking upper bound: %f" %(sum(Best_Ranking[(center_id, g)])/len(Best_Ranking[(center_id, g)]))
				print "matching score by correct match: %f" % (sum(correctMatch[g]) / float(len(correctMatch[g])))
				if compute_sim:
					print "matching score by correct match upper bound %f" % (sum(Best_correctMatch[(center_id, g)]) / float(len(Best_correctMatch[(center_id, g)])))
				if compute_hungarian:
					print "hungarian matching score upper bound: %f" %(sum(hung_score[g])/float(len(hung_score[g])))
				if compute_netalign:
					print "netalign score: %f" %(netalign_scores[g])
				print "percentage of pairs computed: %f" %(this_pair_computed[g]/float(matching_matrix[g].shape[0]*matching_matrix[g].shape[1]))

			if int(metadata['number']) >1:
				derived_matching_matrix = {}
				derived_rank = {}
				derived_netalign = {}
				non_center = matching_matrix.keys()
				for i in xrange(len(non_center)):
					for j in xrange(i+1, len(non_center)):
						derived_matching_matrix[(non_center[i],non_center[j])] = matching_matrix[non_center[i]].T.dot(matching_matrix[non_center[j]])
						Ranking, correct_match = sparseRank(derived_matching_matrix[(non_center[i],non_center[j])], graph_perm[non_center[i]], graph_perm[non_center[j]] , printing=False) # handle!!!!!!!! P1, P2
						derived_rank[(non_center[i],non_center[j])] = sum(Ranking)/len(Ranking)
						derived_netalign[(non_center[i],non_center[j])] = getNetalignScore(multi_graphs[non_center[i]], multi_graphs[non_center[j]], derived_matching_matrix[(non_center[i],non_center[j])])
				print 'derived rank score: '
				print derived_rank
				tmp_avg_derived_rank = sum([v for k,v in derived_rank.iteritems()])/len(derived_rank)
				avg_derived_rank += tmp_avg_derived_rank
				print 'avg derived rank score: ' + str(tmp_avg_derived_rank)
				
				print 'derived netalign score: '
				print derived_netalign
				tmp_avg_netalign = np.mean(derived_netalign.values())
				avg_derived_netalign += tmp_avg_netalign
				print 'avg derived netalign score: ' + str(np.mean(tmp_avg_netalign))	
		
		rank_score /= loop_num * len(pair_count_dict.keys())
		rank_score_upper /= loop_num * len(pair_count_dict.keys())
		correct_score /= loop_num * len(pair_count_dict.keys())
		correct_score_upper /= loop_num * len(pair_count_dict.keys())
		correct_score_hungarian /= loop_num * len(pair_count_dict.keys())
		netalign_score /= loop_num * len(pair_count_dict.keys())
		pairs_computed /= loop_num * len(pair_count_dict.keys())
		avg_derived_rank /= loop_num
		avg_derived_netalign /= loop_num	
		end_matching = time.time()
		matching_time = end_matching - start_matching		

		df = df.append({'filename':filename, 'nodeAttributeFile': metadata['node_dir']\
			, 'noise_level':metadata['noise_level']\
			, 'GraphType':metadata['graph_type'], 'bandNumber':bandNumber, 'adaptiveLSH':adaptiveLSH, 'LSHType':LSHType\
			, 'threshold':threshold\
			, 'rank_score' : rank_score\
			, 'rank_score_upper' : rank_score_upper\
			, 'correct_score' : correct_score\
			, 'correct_score_upper' : correct_score_upper\
			, 'correct_score_hungarian' : correct_score_hungarian\
			, 'center_id': center_id\
			, 'found_center' : metadata['found_center']\
			, 'avg_derived_rank': avg_derived_rank\
			, 'avg_derived_netalign': avg_derived_netalign\
			, 'center_dist': metadata['center_distance']\
			, 'pairs_computed' : pairs_computed\
			# , 'preprocess_time': preprocess_time\
			, 'matching_time': matching_time\
			}, ignore_index=True)

	return df

if __name__ == '__main__':
	adaptiveLSH = [False]
	bandNumber = [4]
	cos_num_plane = [25]
	euc_width = [4]
	LSH = ['Cosine', 'Euclidean']
	folders = ['facebook', 'email']
	# center_distance_types = ['canberra', 'manhattan', 'euclidean']
	fname = 'exp_pair_band_lsh.pkl'

	if os.path.isfile(fname):
		with open(fname, 'rb') as f:
			df = pickle.load(f)
	else:
		df = pd.DataFrame(
			columns=['filename','nodeAttributeFile', 'noise_level', 'GraphType'\
				, 'bandNumber', 'adaptiveLSH', 'LSHType', 'threshold'\
				, 'rank_score', 'rank_score_upper', 'correct_score', 'correct_score_upper', 'correct_score_hungarian'\
				, 'center_id', 'found_center', 'avg_derived_rank', 'center_dist', 'pairs_computed', 'matching_time'])
	# for dist_type in center_distance_types:
	for b in bandNumber:
		for lsh in LSH:
			if lsh == 'Cosine':
				for c in cos_num_plane:
					df = experiment(df, filename = 'facebook', bandNumber = b, adaptiveLSH = False, LSHType = lsh, cos_num_plane = c)
			else:
				for e in euc_width:
					df = experiment(df, filename = 'facebook', bandNumber = b, adaptiveLSH = False, LSHType = lsh, euc_width = e)

		# df = experiment(df, filename = 'Data/phys.edges', nodeAttributeFile = None, 
		# 		GraphType = 'Directed', bandNumber = 2, 
		# 		adaptiveLSH = False, LSHType = 'Cosine', noise_level = 0.001,
		# 		center_distance = dist_type, findcenter = 0)
		# df = experiment(df, filename = 'Data/email.edges', nodeAttributeFile = None, 
		# 		has_noise = True, GraphType = 'Undirected', bandNumber = 2, 
		# 		adaptiveLSH = False, LSHType = 'Cosine', noise_level = 0.01,
		# 		center_distance = dist_type, findcenter = 0)

	pickle.dump(df, open(fname,'wb'))

	writer = pd.ExcelWriter('exp_pair_band_lsh.xlsx')
	df.to_excel(writer, sheet_name='Sheet1')
	writer.save()

