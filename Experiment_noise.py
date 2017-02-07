import matplotlib.pyplot as plt
import numpy as np 
from attr_utils import *
from io_utils import *
from lsh_utils import *
from Experiment import experiment
import pandas as pd
import os.path
import pickle

# def experiment(df, filename = 'metadata/phys.edges', multipleGraph = False, is_perm = False, 
# 	has_noise = False, noise_level = 0.05, 
# 	plotAttribute = False, plotBucket = False, plotCorrectness = False, 
# 	GraphType = 'Directed', bandNumber = 2, adaptiveLSH = True, LSHType = 'Euclidean',
# 	loop_num = 3, cos_num_plane = 20, euc_width = 2):
# 	"""
# 	Experiment on two graphs with multiple setting

# 	"""
# 	path = 'metadata/' + str(GraphType)

# 	A = loadGraph(filename, GraphType)
# 	A = removeIsolatedNodes(A)

# 	if multipleGraph == True:
# 		pass
# 	else:
# 		B, P = permuteNoiseMat(A,is_perm, has_noise, noise_level)

# 	# Write edges to file

# 	writeEdgesToFile(path + '/A.edges', A)
# 	writeEdgesToFile(path + '/B.edges', B)

# 	if GraphType == 'Undirected':
# 		attributesA = getUndirAttribute(path + '/A.edges')

# 		with open(path + '/attributesA', 'w') as f:
# 			for index, row in attributesA.iterrows():
# 				f.write(str(attributesA.ix[index]))

# 		attributesB = getUndirAttribute(path + '/B.edges')

# 		with open(path + '/attributesB', 'w') as f:
# 			for index, row in attributesB.iterrows():
# 				f.write(str(attributesB.ix[index]))


# 		attributes = ['Degree', 'NodeBetweennessCentrality', 'PageRank', 
# 		'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']
# 	else:
# 		attributesA = getDirAttribute(path +'/A.edges')

# 		with open(path+'/attributesA', 'w') as f:
# 			for index, row in attributesA.iterrows():
# 				f.write(str(attributesA.ix[index]))

# 		attributesB = getDirAttribute(path +'/B.edges')

# 		with open(path+'/attributesB', 'w') as f:
# 			for index, row in attributesB.iterrows():
# 				f.write(str(attributesB.ix[index]))

# 		attributes = ['Degree', 'InDegree', 'OutDegree', 'NodeBetweennessCentrality', 
# 					  'PageRank', 'HubsScore', 'AuthoritiesScore',
# 					  'EgonetDegree', 'EgonetInDegree', 'EgonetOutDegree',
# 					  'AvgNeighborDeg', 'AvgNeighborInDeg', 'AvgNeighborOutDeg','EgonetConnectivity']

# 	rank_score = 0
# 	rank_score_upper = 0
# 	correct_score = 0
# 	correct_score_upper = 0
# 	correct_score_hungarian = 0
# 	pairs_computed = 0
# 	for i in range(loop_num):

# 		if GraphType == 'Undirected':
# 			if adaptiveLSH == True :
# 				bandDeg = ['Degree','PageRank','NodeBetweennessCentrality']
# 				bandEdge = ['EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']
					
# 				if LSHType == 'Cosine':
# 					bucketDeg = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandDeg), cos_num_plane)
# 					bucketEdge = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandEdge), cos_num_plane)

# 				elif LSHType == 'Euclidean':
# 					bucketDeg = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandDeg), euc_width)
# 					bucketEdge = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandEdge), euc_width)

# 				buckets = [bucketDeg, bucketEdge]
# 				for i, bucket in enumerate(buckets):
# 					with open(path + '/buckets-band-' + str(i+1), 'w') as f:
# 						for k, v in bucket.items():
# 							f.write(str(k) + str(v) + '\n')
				
# 			else:
# 				band_all = list(attributes)
# 				np.random.shuffle(band_all)
# 				randomBand = [band_all[i*len(band_all)/bandNumber: (i + 1)*len(band_all)/bandNumber]for i in range(bandNumber)]

# 				buckets = []

# 				if LSHType == 'Cosine':
# 					for band in randomBand:
# 						buckets.append(generateCosineBuckets(selectAndCombine(attributesA, attributesB, band), cos_num_plane))


# 				elif LSHType == 'Euclidean':
# 					for band in randomBand:
# 						buckets.append(generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, band), euc_width))

# 				for i, bucket in enumerate(buckets):
# 					with open(path + '/buckets-band-' + str(i+1), 'w') as f:
# 						for k, v in bucket.items():
# 							f.write(str(k) + str(v) + '\n')

		
# 		elif GraphType == 'Directed':

# 			if adaptiveLSH == True:
# 				bandDeg = ['Degree','InDegree','OutDegree']
# 				bandEgo = ['EgonetDegree', 'EgonetInDegree', 'EgonetOutDegree',
# 						  'AvgNeighborDeg', 'AvgNeighborInDeg', 'AvgNeighborOutDeg','EgonetConnectivity']
# 				bandCentr = ['PageRank', 'NodeBetweennessCentrality', 
# 							 'HubsScore', 'AuthoritiesScore']

# 				if LSHType == 'Cosine':

# 					bucketDeg = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandDeg), cos_num_plane)
# 					bucketEgo = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandEgo), cos_num_plane)
# 					bucketCentr = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandCentr), cos_num_plane)

# 				elif LSHType == 'Euclidean':
# 					bucketDeg = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandDeg), euc_width)
# 					bucketEgo = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandEgo), euc_width)
# 					bucketCentr = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandCentr), euc_width)

# 				buckets = [bucketDeg, bucketEgo, bucketCentr]
# 				for i, bucket in enumerate(buckets):
# 					with open(path+'/buckets-band'+str(i+1), 'w') as f:
# 						for k, v in bucket.items():
# 							f.write(str(k) + str(v) + '\n')

# 			else:
# 				band_all = list(attributes)
# 				np.random.shuffle(band_all)
# 				randomBand = [band_all[i*len(band_all)/bandNumber: (i + 1)*len(band_all)/bandNumber]for i in range(bandNumber)]

# 				buckets = []

# 				if LSHType == 'Cosine':
# 					for band in randomBand:
# 						buckets.append(generateCosineBuckets(selectAndCombine(attributesA, attributesB, band), cos_num_plane))


# 				elif LSHType == 'Euclidean':
# 					for band in randomBand:
# 						buckets.append(generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, band), euc_width))

# 				for i, bucket in enumerate(buckets):
# 					with open(path+'/buckets-band-'+str(i+1), 'w') as f:
# 						for k, v in bucket.items():
# 							f.write(str(k) + str(v) + '\n')

# 		combineAB = selectAndCombine(attributesA, attributesB)	 
# 		pair_count_dict = combineBucketsBySum(buckets, combineAB, path+'/A.edges')
# 		matching_matrix = computeMatchingMat(attributesA, attributesB, pair_count_dict, LSHType, threshold=1)
# 		sim_matrix = computeWholeSimMat(attributesA, attributesB, LSHType)

# 		Ranking = Rank(matching_matrix, P)
# 		Best_ranking = Rank(sim_matrix, P)
# 		correctMatch = argmaxMatch(matching_matrix, attributesA, attributesB, P)
# 		Best_correctMatch = argmaxMatch(sim_matrix, attributesA, attributesB, P)
# 		hung_score = hungarianMatch(sim_matrix, P)

# 		rank_score += sum(Ranking)/len(Ranking)
# 		rank_score_upper += sum(Ranking)/len(Ranking)
# 		correct_score += sum(correctMatch) / float(len(correctMatch))
# 		correct_score_upper += sum(Best_correctMatch) / float(len(Best_correctMatch))
# 		correct_score_hungarian += sum(hung_score)/float(len(hung_score))
# 		pairs_computed += len(pair_count_dict)/float(matching_matrix.shape[0]*matching_matrix.shape[1])

# 		print "=========================================================="
# 		print "is_perm = " + str(is_perm) + ", has_noise = "+ str(has_noise)+", noise_level = "+ str(noise_level)+", GraphType = "+ GraphType
# 		print "bandNumber = "+str(bandNumber)+", adaptiveLSH = "+ str(adaptiveLSH)+", LSHType = "+LSHType
# 		print "matching score by ranking: %f" %(sum(Ranking)/len(Ranking))
# 		print "matching score by ranking upper bound: %f" %(sum(Ranking)/len(Ranking))
# 		print "matching score by correct match: %f" % (sum(correctMatch) / float(len(correctMatch)))
# 		print "matching score by correct match upper bound %f" % (sum(Best_correctMatch) / float(len(Best_correctMatch)))
# 		print "hungarian matching score upper bound: %f" %(sum(hung_score)/float(len(hung_score)))
# 		print "percentage of pairs computed: %f" %(len(pair_count_dict)/float(matching_matrix.shape[0]*matching_matrix.shape[1]))

# 	rank_score /= loop_num
# 	rank_score_upper /= loop_num
# 	correct_score /= loop_num
# 	correct_score_upper /= loop_num
# 	correct_score_hungarian /= loop_num
# 	pairs_computed /= loop_num


# 	df = df.append({'filename':filename, 'is_perm':is_perm, 'has_noise':has_noise, 'noise_level':noise_level\
# 		, 'GraphType':GraphType, 'bandNumber':bandNumber, 'adaptiveLSH':adaptiveLSH, 'LSHType':LSHType\
# 		, 'rank_score' : rank_score\
# 		, 'rank_score_upper' : rank_score_upper\
# 		, 'correct_score' : correct_score\
# 		, 'correct_score_upper' : correct_score_upper\
# 		, 'correct_score_hungarian' : correct_score_hungarian\
# 		, 'pairs_computed' : pairs_computed\
# 		}, ignore_index=True)

# 	return df


adaptiveLSH = [False]
noise = [True]
bandNumber = [2,4,8]
fname = 'exp_result_noise.pkl'

if os.path.isfile(fname):
	with open(fname, 'rb') as f:
		df = pickle.load(f)
else:
	df = pd.DataFrame(
		columns=['filename', 'is_perm', 'has_noise', 'noise_level', 'GraphType', 'bandNumber', 'adaptiveLSH', 'LSHType'\
			, 'rank_score', 'rank_score_upper', 'correct_score', 'correct_score_upper', 'correct_score_hungarian'\
			, 'pairs_computed'])


a = True
n = True
b = 2
noise_level = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.1, 0.12, 0.16]
for a in adaptiveLSH:
	for n in noise:
		for b in bandNumber:
			for l in noise_level:
				df = experiment(df, filename = 'metadata/phys.edges', multipleGraph = False, is_perm = False, 
					has_noise = n, noise_level = l, plotAttribute = False, plotBucket = False, plotCorrectness = False, 
					GraphType = 'Directed', bandNumber = b, adaptiveLSH = a, LSHType = 'Cosine')
				df = experiment(df, filename = 'metadata/phys.edges', multipleGraph = False, is_perm = False, 
					has_noise = n, noise_level = l, plotAttribute = False, plotBucket = False, plotCorrectness = False, 
					GraphType = 'Directed', bandNumber = b, adaptiveLSH = a, LSHType = 'Euclidean')



pickle.dump(df, open(fname,'wb'))

# write to excel
writer = pd.ExcelWriter('exp_result_noise.xlsx')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()

#plot result
df_cos = df[df['LSHType']=='Cosine'].loc[:,['noise_level','rank_score', 'correct_score_hungarian',\
										'pairs_computed']]
x_cos = df_cos['noise_level'].values
rank_cos = df_cos['rank_score'].values
hung_cos = df_cos['correct_score_hungarian'].values
pair_cos = df_cos['pairs_computed'].values

line1, = plt.plot(x_cos, rank_cos, label="rank_score", linestyle='--')
line2, = plt.plot(x_cos, hung_cos, label="correct score hungarian", linewidth=2)
line3, = plt.plot(x_cos, pair_cos, label="% pairs computed", linewidth=2)
plt.legend(handles=[line1, line2, line3])
plt.suptitle('Accuracy to Noise(Cosine LSH)')
plt.show()

df_euc = df[df['LSHType']=='Euclidean'].loc[:,['noise_level','rank_score', 'correct_score_hungarian',\
										'pairs_computed']]
x_euc = df_euc['noise_level'].values
rank_euc = df_euc['rank_score'].values
hung_euc = df_euc['correct_score_hungarian'].values
pair_euc = df_euc['pairs_computed'].values

line1, = plt.plot(x_euc, rank_euc, label="rank_score", linestyle='--')
line2, = plt.plot(x_euc, hung_euc, label="correct score hungarian", linewidth=2)
line3, = plt.plot(x_euc, pair_euc, label="% pairs computed", linewidth=2)
plt.legend(handles=[line1, line2, line3])
plt.suptitle('Accuracy to Noise(Euclidean LSH)')
plt.show()
