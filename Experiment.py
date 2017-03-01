import matplotlib.pyplot as plt
import numpy as np 
from attr_utils import *
from io_utils import *
from lsh_utils import *
import pandas as pd
import os.path
import pickle
import time


def experiment(df, filename = 'metadata/phys.edges', nodeAttributeFile = None, multipleGraph = False, is_perm = False, 
	has_noise = False, noise_level = 0.05, plotAttribute = False, plotBucket = False, plotCorrectness = False, 
	GraphType = 'Directed', bandNumber = 2, adaptiveLSH = True, LSHType = 'Euclidean',
	loop_num = 3, cos_num_plane = 25, euc_width = 2, compute_hungarian = True, compute_sim = True,
	threshold = 1):
	"""
	Experiment on two graphs with multiple setting

	"""
	start_preprocess = time.time()
	path = 'metadata/' + str(GraphType)

	A = loadGraph(filename, GraphType)
	A, rest_idx = removeIsolatedNodes(A)

	if nodeAttributeFile is not None:
		nodeAttributesValue, nodeAttributesName = loadNodeFeature(nodeAttributeFile)
		nodeAttributesValue = [nodeAttributesValue[i] for i in rest_idx]
	else:
		nodeAttributesValue, nodeAttributesName = [], []

	if multipleGraph == True:
		pass
	else:
		B, P = permuteNoiseMat(A,is_perm, has_noise, noise_level)
	# Write edges to file
	writeEdgesToFile(path+'/A.edges', A)
	writeEdgesToFile(path+'/B.edges', B)

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
	if compute_sim:
		sim_matrix = computeWholeSimMat(attributesA, attributesB, LSHType)

	rank_score = 0
	rank_score_upper = 0
	correct_score = 0
	correct_score_upper = 0
	correct_score_hungarian = 0
	pairs_computed = 0
	matching_time = 0
	for i in range(loop_num):
		start_matching = time.time()
		if GraphType == 'Undirected':
			if adaptiveLSH == True :
				bandDeg = ['Degree','PageRank','NodeBetweennessCentrality']
				bandEdge = ['EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']
				bandNode = nodeAttributesName[:]
					
				if LSHType == 'Cosine':
					bucketDeg = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandDeg), cos_num_plane)
					bucketEdge = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandEdge), cos_num_plane)

				elif LSHType == 'Euclidean':
					bucketDeg = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandDeg), euc_width)
					bucketEdge = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandEdge), euc_width)

				bucketNode = generateNodeBuckets(LSHType, attributesA, attributesB, bandNode)
				buckets = [bucket for bucket in [bucketDeg, bucketEdge, bucketNode] if len(bucket) > 0]

				for i, bucket in enumerate(buckets):
					with open(path + '/buckets-band-' + str(i+1), 'w') as f:
						for k, v in bucket.items():
							f.write(str(k) + str(v) + '\n')
			else:
				band_all = list(attributes)
				np.random.shuffle(band_all)
				randomBand = [band_all[i*len(band_all)/bandNumber: (i + 1)*len(band_all)/bandNumber]for i in range(bandNumber)]

				buckets = []

				if LSHType == 'Cosine':
					for band in randomBand:
						buckets.append(generateCosineBuckets(selectAndCombine(attributesA, attributesB, band), cos_num_plane))


				elif LSHType == 'Euclidean':
					for band in randomBand:
						buckets.append(generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, band), euc_width))

				for i, bucket in enumerate(buckets):
					with open(path + '/buckets-band-' + str(i+1), 'w') as f:
						for k, v in bucket.items():
							f.write(str(k) + str(v) + '\n')

		
		elif GraphType == 'Directed':
			if adaptiveLSH == True:
				bandDeg = ['Degree','InDegree','OutDegree']
				bandEgo = ['EgonetDegree', 'EgonetInDegree', 'EgonetOutDegree',
						  'AvgNeighborDeg', 'AvgNeighborInDeg', 'AvgNeighborOutDeg','EgonetConnectivity']
				bandCentr = ['PageRank', 'NodeBetweennessCentrality', 
							 'HubsScore', 'AuthoritiesScore']
				bandNode = nodeAttributesName[:]

				if LSHType == 'Cosine':

					bucketDeg = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandDeg), cos_num_plane)
					bucketEgo = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandEgo), cos_num_plane)
					bucketCentr = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandCentr), cos_num_plane)

				elif LSHType == 'Euclidean':
					bucketDeg = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandDeg), euc_width)
					bucketEgo = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandEgo), euc_width)
					bucketCentr = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandCentr), euc_width)

				bucketNode = generateNodeBuckets(LSHType, attributesA, attributesB, bandNode)
				buckets = [bucket for bucket in [bucketDeg, bucketEgo, bucketCentr, bucketNode] if len(bucket) > 0]

				for i, bucket in enumerate(buckets):
					with open(path+'/buckets-band'+str(i+1), 'w') as f:
						for k, v in bucket.items():
							f.write(str(k) + str(v) + '\n')

			else:
				band_all = list(attributes)
				np.random.shuffle(band_all)
				randomBand = [band_all[i*len(band_all)/bandNumber: (i + 1)*len(band_all)/bandNumber]for i in range(bandNumber)]

				buckets = []

				if LSHType == 'Cosine':
					for band in randomBand:
						buckets.append(generateCosineBuckets(selectAndCombine(attributesA, attributesB, band), cos_num_plane))


				elif LSHType == 'Euclidean':
					for band in randomBand:
						buckets.append(generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, band), euc_width))

				for i, bucket in enumerate(buckets):
					with open(path+'/buckets-band-'+str(i+1), 'w') as f:
						for k, v in bucket.items():
							f.write(str(k) + str(v) + '\n')

		combineAB = selectAndCombine(attributesA, attributesB)	 
		pair_count_dict = combineBucketsBySum(buckets, combineAB, path+'/A.edges')
		matching_matrix, this_pair_computed = computeMatchingMat(attributesA, attributesB, pair_count_dict, LSHType, threshold)
		

		Ranking = Rank(matching_matrix, P)
		Best_Ranking = Ranking
		if compute_sim:
			Best_Ranking = Rank(sim_matrix, P)
		
		correctMatch = argmaxMatch(matching_matrix, attributesA, attributesB, P)
		Best_correctMatch = correctMatch
		if compute_sim:
			Best_correctMatch = argmaxMatch(sim_matrix, attributesA, attributesB, P)
		hung_score = correctMatch
		if compute_hungarian:
			hung_score = hungarianMatch(sim_matrix, P)
		
		end_matching = time.time()
		matching_time += end_matching - start_matching

		rank_score += sum(Ranking)/len(Ranking)
		if compute_sim:
			rank_score_upper += sum(Best_Ranking)/len(Best_Ranking)
			correct_score_upper += sum(Best_correctMatch) / float(len(Best_correctMatch))
		else:
			rank_score_upper += 0
			correct_score_upper += 0
		correct_score += sum(correctMatch) / float(len(correctMatch))
		if compute_hungarian:
			correct_score_hungarian += sum(hung_score)/float(len(hung_score))
		else:
			correct_score_hungarian += 0
		pairs_computed += this_pair_computed/float(matching_matrix.shape[0]*matching_matrix.shape[1])

		print "=========================================================="
		print filename
		print "is_perm = " + str(is_perm) + ", has_noise = "+ str(has_noise)+", GraphType = "+ GraphType
		print "bandNumber = "+str(bandNumber)+", adaptiveLSH = "+ str(adaptiveLSH)+", LSHType = "+LSHType
		print "noise_level = "+str(noise_level)+", nodeAttributeFile = "+str(nodeAttributeFile)+", threshold = "+str(threshold)
		print "matching score by ranking: %f" %(sum(Ranking)/len(Ranking))
		if compute_sim:
			print "matching score by ranking upper bound: %f" %(sum(Best_Ranking)/len(Best_Ranking))
		print "matching score by correct match: %f" % (sum(correctMatch) / float(len(correctMatch)))
		if compute_sim:
			print "matching score by correct match upper bound %f" % (sum(Best_correctMatch) / float(len(Best_correctMatch)))
		if compute_hungarian:
			print "hungarian matching score upper bound: %f" %(sum(hung_score)/float(len(hung_score)))
		print "percentage of pairs computed: %f" %(this_pair_computed/float(matching_matrix.shape[0]*matching_matrix.shape[1]))

	rank_score /= loop_num
	rank_score_upper /= loop_num
	correct_score /= loop_num
	correct_score_upper /= loop_num
	correct_score_hungarian /= loop_num
	pairs_computed /= loop_num
	matching_time /= loop_num

	df = df.append({'filename':filename, 'nodeAttributeFile': str(nodeAttributeFile), 'is_perm':is_perm\
		, 'has_noise':has_noise, 'noise_level':noise_level\
		, 'GraphType':GraphType, 'bandNumber':bandNumber, 'adaptiveLSH':adaptiveLSH, 'LSHType':LSHType\
		, 'threshold':threshold\
		, 'rank_score' : rank_score\
		, 'rank_score_upper' : rank_score_upper\
		, 'correct_score' : correct_score\
		, 'correct_score_upper' : correct_score_upper\
		, 'correct_score_hungarian' : correct_score_hungarian\
		, 'pairs_computed' : pairs_computed\
		, 'preprocess_time': preprocess_time\
		, 'matching_time': matching_time\
		}, ignore_index=True)

	if plotAttribute == True:
		

		plt.clf()
		for attr in attributes:
			print attr
			plt.figure()
			bins = np.linspace(min(min(attributesA[attr]), min(attributesB[attr])), max(max(attributesA[attr]), max(attributesB[attr])), 40)
			plt.hist(attributesA[attr], bins, alpha = 0.5, label = 'Origin')
			plt.hist(attributesB[attr], bins, alpha = 0.5, label = 'With noise')
			plt.xlabel(attr)
			plt.ylabel('Frequency')
			plt.legend(loc = 'best')
			plt.show()

	if plotBucket == True:
		plt.clf()
		f = plt.figure()
		for i, bucket in enumerate(buckets):
			f.add_subplot(len(buckets), 1, i+1)
			plotBucketDistribution(bucket)

		plt.show()

	if plotCorrectness == True:
		if LSHType == 'Cosine':
			bucketAll = generateCosineBuckets(selectAndCombine(attributesA, attributesB, attributes), 20)
		elif LSHType == 'Euclidean':
			bucketAll = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, attributes), 20)
		for bucket in buckets:
			plotBucketCorrectness(bucket, attributesA.shape[0])

		plotCorrectness(bucketAll, attributesA.shape[0])

	return df

if __name__ == '__main__':
	adaptiveLSH = [False]
	noise = [True]
	bandNumber = [2,4,8]
	fname = 'exp_result_attr.pkl'

	if os.path.isfile(fname):
		with open(fname, 'rb') as f:
			df = pickle.load(f)
	else:
		df = pd.DataFrame(
			columns=['filename','nodeAttributeFile', 'is_perm', 'has_noise', 'GraphType'\
				, 'bandNumber', 'adaptiveLSH', 'LSHType', 'threshold'\
				, 'rank_score', 'rank_score_upper', 'correct_score', 'correct_score_upper', 'correct_score_hungarian'\
				, 'pairs_computed'])


	for a in adaptiveLSH:
		for n in noise:
			for b in bandNumber:
				df = experiment(df, filename = 'metadata/A.edges', nodeAttributeFile = None, 
					multipleGraph = False, is_perm = False, has_noise = n, plotAttribute = False, 
					plotBucket = False, plotCorrectness = False, GraphType = 'Undirected', bandNumber = b, 
					adaptiveLSH = a, LSHType = 'Cosine')
				df = experiment(df, filename = 'metadata/A.edges', nodeAttributeFile = None, 
					multipleGraph = False, is_perm = False, has_noise = n, plotAttribute = False, 
					plotBucket = False, plotCorrectness = False, GraphType = 'Undirected', bandNumber = b, 
					adaptiveLSH = a, LSHType = 'Euclidean')

				df = experiment(df, filename = 'metadata/phys.edges', nodeAttributeFile = None, 
					multipleGraph = False, is_perm = False, has_noise = n, plotAttribute = False, 
					plotBucket = False, plotCorrectness = False, GraphType = 'Directed', bandNumber = b, 
					adaptiveLSH = a, LSHType = 'Cosine')
				df = experiment(df, filename = 'metadata/phys.edges', nodeAttributeFile = None, 
					multipleGraph = False, is_perm = False, has_noise = n, plotAttribute = False, 
					plotBucket = False, plotCorrectness = False, GraphType = 'Directed', bandNumber = b, 
					adaptiveLSH = a, LSHType = 'Euclidean')
				if a:
					break


	#df = experiment(df, filename = 'metadata/phys.edges', nodeAttributeFile = 'phys.nodes', multipleGraph = False, is_perm = False, 
	#                has_noise = True, plotAttribute = False, plotBucket = False, plotCorrectness = False, 
	#                GraphType = 'Directed', bandNumber = 2, adaptiveLSH = True, LSHType = 'Cosine')

	pickle.dump(df, open(fname,'wb'))

	# experiment(df, filename = 'facebook/0.edges', multipleGraph = False, is_perm = False, 
	#           has_noise = True, plotAttribute = False, plotBucket = False, plotCorrectness = False, 
	#           GraphType = 'Undirected', bandNumber = 4, adaptiveLSH = False, LSHType = 'Cosine')

	writer = pd.ExcelWriter('exp_result_attr.xlsx')
	df.to_excel(writer, sheet_name='Sheet1')
	writer.save()