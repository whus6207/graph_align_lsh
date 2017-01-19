import matplotlib.pyplot as plt
import numpy as np 
from attr_utils import *
from io_utils import *
from lsh_utils import *
import pandas as pd
import os.path
import pickle


def experiment(df, filename = 'metadata/phys.edges', multipleGraph = False, is_perm = False, 
	has_noise = False, plotAttribute = False, plotBucket = False, plotCorrectness = False, GraphType = 'Directed', bandNumber = 2, adaptiveLSH = True, LSHType = 'Euclidean'):
	"""
	Experiment on two graphs with multiple setting

	"""
	path = 'metadata/' + str(GraphType)

	A = loadGraph(filename, GraphType)
	A = removeIsolatedNodes(A)

	if multipleGraph == True:
		pass
	else:
		B, P = permuteNoiseMat(A,is_perm, has_noise)

	# Write edges to file

	writeEdgesToFile(path + '/A.edges', A)
	writeEdgesToFile(path + '/B.edges', B)


	if GraphType == 'Undirected':
		attributesA = getUndirAttribute(path + '/A.edges')

		with open(path + '/attributesA', 'w') as f:
			for index, row in attributesA.iterrows():
				f.write(str(attributesA.ix[index]))

		attributesB = getUndirAttribute(path + '/B.edges')

		with open(path + '/attributesB', 'w') as f:
			for index, row in attributesB.iterrows():
				f.write(str(attributesB.ix[index]))


		attributes = ['Degree', 'NodeBetweennessCentrality', 'PageRank', 
		'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']


		if adaptiveLSH == True :
			bandDeg = ['Degree','PageRank','NodeBetweennessCentrality']
			bandEdge = ['EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']
				
			if LSHType == 'Cosine':
				bucketDeg = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandDeg), 20)
				bucketEdge = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandEdge), 20)

			elif LSHType == 'Euclidean':
				bucketDeg = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandDeg), 2)
				bucketEdge = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandEdge), 2)

			buckets = [bucketDeg, bucketEdge]
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
					buckets.append(generateCosineBuckets(selectAndCombine(attributesA, attributesB, band), 20))


			elif LSHType == 'Euclidean':
				for band in randomBand:
					buckets.append(generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, band), 2))

			for i, bucket in enumerate(buckets):
				with open(path + '/buckets-band-' + str(i+1), 'w') as f:
					for k, v in bucket.items():
						f.write(str(k) + str(v) + '\n')

	
	elif GraphType == 'Directed':
		attributesA = getDirAttribute(path +'/A.edges')

		with open(path+'/attributesA', 'w') as f:
			for index, row in attributesA.iterrows():
				f.write(str(attributesA.ix[index]))

		attributesB = getDirAttribute(path +'/B.edges')

		with open(path+'/attributesB', 'w') as f:
			for index, row in attributesB.iterrows():
				f.write(str(attributesB.ix[index]))

		attributes = ['Degree', 'InDegree', 'OutDegree', 'NodeBetweennessCentrality', 
					  'PageRank', 'HubsScore', 'AuthoritiesScore',
					  'EgonetDegree', 'EgonetInDegree', 'EgonetOutDegree',
					  'AvgNeighborDeg', 'AvgNeighborInDeg', 'AvgNeighborOutDeg','EgonetConnectivity']

		if adaptiveLSH == True:
			bandDeg = ['Degree','InDegree','OutDegree']
			bandEgo = ['EgonetDegree', 'EgonetInDegree', 'EgonetOutDegree',
					  'AvgNeighborDeg', 'AvgNeighborInDeg', 'AvgNeighborOutDeg','EgonetConnectivity']
			bandCentr = ['PageRank', 'NodeBetweennessCentrality', 
						 'HubsScore', 'AuthoritiesScore']

			if LSHType == 'Cosine':

				bucketDeg = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandDeg), 20)
				bucketEgo = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandEgo), 20)
				bucketCentr = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandCentr), 20)

			elif LSHType == 'Euclidean':
				bucketDeg = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandDeg), 2)
				bucketEgo = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandEgo), 2)
				bucketCentr = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandCentr), 2)

			buckets = [bucketDeg, bucketEgo, bucketCentr]
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
					buckets.append(generateCosineBuckets(selectAndCombine(attributesA, attributesB, band), 20))


			elif LSHType == 'Euclidean':
				for band in randomBand:
					buckets.append(generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, band), 2))

			for i, bucket in enumerate(buckets):
				with open(path+'/buckets-band-'+str(i+1), 'w') as f:
					for k, v in bucket.items():
						f.write(str(k) + str(v) + '\n')

	combineAB = selectAndCombine(attributesA, attributesB)	 
	pair_count_dict = combineBucketsBySum(buckets, combineAB, path+'/A.edges')
	matching_matrix = computeMatchingMat(attributesA, attributesB, pair_count_dict, LSHType, threshold=1)
	sim_matrix = computeWholeSimMat(attributesA, attributesB, LSHType)

	Ranking = Rank(matching_matrix, P)
	Best_ranking = Rank(sim_matrix, P)
	correctMatch = argmaxMatch(matching_matrix, attributesA, attributesB, P)
	Best_correctMatch = argmaxMatch(sim_matrix, attributesA, attributesB, P)



	print "=========================================================="
	print "is_perm = " + str(is_perm) + ", has_noise = "+ str(has_noise)+", GraphType = "+ GraphType
	print "bandNumber = "+str(bandNumber)+", adaptiveLSH = "+ str(adaptiveLSH)+", LSHType = "+LSHType
	print "matching score by ranking: %f" %(sum(Ranking)/len(Ranking))
	print "matching score by ranking upper bound: %f" %(sum(Best_ranking)/len(Best_ranking))
	score = hungarianMatch(sim_matrix, P)
	upper_bound_score = sum(score)/float(len(score))
	print "hungarian matching score upper bound: %f" %(upper_bound_score)
	print "matching score by correct match: %f" % (sum(correctMatch) / float(len(correctMatch)))
	print "matching score by correct match upper bound %f" % (sum(Best_correctMatch) / float(len(Best_correctMatch)))
	print "percentage of pairs computed: %f" %(len(pair_count_dict)/float(matching_matrix.shape[0]*matching_matrix.shape[1]))
	df = df.append({'filename':filename, 'is_perm':is_perm, 'has_noise':has_noise, 'GraphType':GraphType\
		, 'bandNumber':bandNumber, 'adaptiveLSH':adaptiveLSH, 'LSHType':LSHType\
		, 'rank_score' : sum(Ranking)/len(Ranking)\
		, 'rank_score_upper' : sum(Best_ranking)/len(Best_ranking)\
		, 'correct_score' : (sum(correctMatch) / float(len(correctMatch)))\
		, 'correct_score_upper' : (sum(Best_correctMatch) / float(len(Best_correctMatch)))\
		, 'correct_score_hungarian' : upper_bound_score\
		, 'pairs_computed' : (len(pair_count_dict)/float(matching_matrix.shape[0]*matching_matrix.shape[1]))\
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


adaptiveLSH = [True, False]
noise = [True, False]
bandNumber = [2,4,8]
fname = 'exp_result.pkl'

if os.path.isfile(fname):
	with open(fname, 'rb') as f:
		df = pickle.load(f)
else:
	df = pd.DataFrame(
		columns=['filename', 'is_perm', 'has_noise', 'GraphType', 'bandNumber', 'adaptiveLSH', 'LSHType'\
			, 'rank_score', 'rank_score_upper', 'correct_score', 'correct_score_upper', 'correct_score_hungarian'\
			, 'pairs_computed'])

for a in adaptiveLSH:
	for n in noise:
		for b in bandNumber:
			df = experiment(df, filename = 'metadata/A.edges', multipleGraph = False, is_perm = False, 
				has_noise = n, plotAttribute = False, plotBucket = False, plotCorrectness = False, 
				GraphType = 'Undirected', bandNumber = b, adaptiveLSH = a, LSHType = 'Cosine')
			df = experiment(df, filename = 'metadata/A.edges', multipleGraph = False, is_perm = False, 
				has_noise = n, plotAttribute = False, plotBucket = False, plotCorrectness = False, 
				GraphType = 'Undirected', bandNumber = b, adaptiveLSH = a, LSHType = 'Euclidean')

			df = experiment(df, filename = 'metadata/phys.edges', multipleGraph = False, is_perm = False, 
				has_noise = n, plotAttribute = False, plotBucket = False, plotCorrectness = False, 
				GraphType = 'Directed', bandNumber = b, adaptiveLSH = a, LSHType = 'Cosine')
			df = experiment(df, filename = 'metadata/phys.edges', multipleGraph = False, is_perm = False, 
				has_noise = n, plotAttribute = False, plotBucket = False, plotCorrectness = False, 
				GraphType = 'Directed', bandNumber = b, adaptiveLSH = a, LSHType = 'Euclidean')
			if a:
				break


pickle.dump(df, open(fname,'wb'))
# experiment(df, filename = 'facebook/0.edges', multipleGraph = False, is_perm = False, 
# 			has_noise = True, plotAttribute = False, plotBucket = False, plotCorrectness = False, 
#  			GraphType = 'Undirected', bandNumber = 4, adaptiveLSH = False, LSHType = 'Cosine')

# writer = pd.ExcelWriter('exp_result.xlsx')
# df.to_excel(writer, sheet_name='Sheet1')
# writer.save()