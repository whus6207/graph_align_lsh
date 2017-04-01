import numpy as np
from collections import defaultdict
from scipy.stats import entropy
from scipy.sparse import lil_matrix
from itertools import izip
import matplotlib.pyplot as plt
import random
from munkres import Munkres

def generateCosineBuckets(attributes, cols):
    attr = attributes.drop(['Graph', 'Id'], axis=1).as_matrix()
    randMatrix = np.random.normal(size=(attr.shape[1], cols))
    #randMatrix = np.multiply(((randMatrix - 0.5) * 2).T, 1 / np.median(attributes, axis=0)).T
    randMatrix = np.multiply(((randMatrix - 0.5) * 2).T, 1 / np.mean(attr, axis=0)).T
    #randMatrix = (randMatrix - 0.5) * 2
    signMatrix = attr.dot(randMatrix)
    signMatrix[signMatrix > 0] = 1
    signMatrix[signMatrix < 0] = 0
    hashMatrix = signMatrix.dot([2**i for i in range(cols)])
    dic = defaultdict(list)
    for i in range(len(hashMatrix)):
        dic[hashMatrix[i]].append((attributes.ix[i]['Graph'], attributes.ix[i]['Id']))
    return dic

def generateEuclideanBuckets(attributes, bin_wid):
    attr = attributes.drop(['Graph', 'Id'], axis=1).as_matrix()
    randVec = np.random.normal(size=(attr.shape[1],))
    randVec = np.multiply((randVec).T, 10 / np.mean(attr, axis=0)).T
    bias = random.uniform(0, bin_wid)
    
    hashVec = (attr.dot(randVec)+bias)/bin_wid
    hashVec = hashVec.astype(int)
    dic = defaultdict(list)
    for i in range((len(hashVec))):
        dic[hashVec[i]].append((attributes.ix[i]['Graph'], attributes.ix[i]['Id']))
    return dic

def generateNodeBuckets(LSHType, attributesA, attributesB, bandNode):
    bucketNode = []
    if len(bandNode) > 0:
        if LSHType == 'Cosine':
            bucketNode = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandNode), 20)
        elif LSHType == 'Euclidean':
            bucketNode = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandNode), 20)
    return bucketNode

def generateNodeBucketsMulti(LSHType, graph_attrs, bandNode, cos_num_plane, euc_width):
    bucketNode = []
    if len(bandNode) > 0:
        if LSHType == 'Cosine':
            bucketNode = generateCosineBuckets(selectAndCombineMulti(graph_attrs, bandNode), cos_num_plane)
        elif LSHType == 'Euclidean':
            bucketNode = generateEuclideanBuckets(selectAndCombineMulti(graph_attrs, bandNode), euc_width)
    return bucketNode

def selectAndCombine(A, B, cols = None):
    if cols is not None:
        return A[cols + ['Graph', 'Id']].append(B[cols + ['Graph', 'Id']], ignore_index=True)
        #return np.vstack((A[cols].as_matrix(), B[cols].as_matrix()))
    else:
        #return np.vstack((A.as_matrix(), B.as_matrix()))
        return A.append(B, ignore_index=True)

def selectAndCombineMulti(graph_attrs, cols = None):
    graphs = graph_attrs.keys()
    if cols is not None:
        stacked_attr = graph_attrs[graphs[0]][cols + ['Graph', 'Id']]
    else:
        stacked_attr = graph_attrs[graphs[0]]
    for i in xrange(1, len(graphs)):
        if cols is not None:
            stacked_attr = stacked_attr.append(graph_attrs[graphs[i]][cols + ['Graph', 'Id']], ignore_index=True)
        else:
            stacked_attr = stacked_attr.append(graph_attrs[graphs[i]], ignore_index=True)
    return stacked_attr

def cos_sim(v1, v2, scaling=None):
    if scaling is None:
        scaling = np.ones((len(v1),))
    v1 = np.multiply(v1, 1/scaling)
    v2 = np.multiply(v2, 1/scaling)
    return v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def KL_sim(distribution1, distribution2):
    v1, bin1 = np.histogram(distribution1, 30)
    v2, bin2 = np.histogram(distribution2, 30)
    v1 = v1.astype(float)
    v2 = v2.astype(float)
    v1 = v1 / sum(v1)
    v2 = v2 / sum(v2) 
    return np.sum(np.where((v1 != 0) & (v2 !=0), v1 * np.log(v1 / v2), 0), axis = 0)

def Euclidean_sim(v1,v2, scaling = None):
    if scaling is None:
        scaling = np.ones((len(v1, )))
    assert (len(v1) == len(v2)), "Dimension is different"
    v1 = np.multiply(v1, 1 / scaling)
    v2 = np.multiply(v2, 1 / scaling)
    eucDis = sum((v1 - v2) ** 2) ** 0.5
    return 1/(1+eucDis)


def computeEulideanMatchingMat(attributesA, attributesB, pair_count_dict):
    matching_matrix = np.zeros((len(attributesA), len(attributesB)))
    for i in range(len(a)):
        for j in range(len(b)):
            matching_matrix[i,j] = Euclidean_Sim(a[i],b[j])
    return retMat

def computeMatchingMat(attributesA, attributesB, pair_count_dict, LSHType, threshold = 1):
    combineAB = selectAndCombine(attributesA, attributesB)
    combineAB = combineAB.as_matrix()
    matching_matrix = np.zeros((len(attributesA), len(attributesB)))
    scale = np.mean(combineAB[:,2:], axis=0) # Still taking mean?
    pair_computed = 0
    if LSHType == 'Cosine':               
        for pair, count in pair_count_dict.items():
            if count >= threshold:
                pair_computed += 1
                matching_matrix[pair[0]][pair[1]] = cos_sim(combineAB[pair[0]][2:],\
                    combineAB[pair[1]+len(attributesA)][2:],scaling=scale)*count
    elif LSHType == 'Euclidean':
        for pair, count in pair_count_dict.items():
            if count >= threshold:
                pair_computed += 1
                matching_matrix[pair[0]][pair[1]] = Euclidean_sim(combineAB[pair[0]][2:],\
                    combineAB[pair[1]+len(attributesA)][2:],scaling=scale)*count
        
    return matching_matrix, pair_computed

def computeSparseMatchingMat(attributesA, attributesB, pair_count_dict, LSHType, threshold = 1):
    combineAB = selectAndCombine(attributesA, attributesB)
    matching_matrix = lil_matrix((len(attributesA), len(attributesB)))
    scale = np.mean(combineAB[:,2:], axis=0)
    pair_computed = 0
    if LSHType == 'Cosine':               
        for pair, count in pair_count_dict.items():
            if count >= threshold:
                pair_computed += 1
                matching_matrix[pair[0], pair[1]-len(attributesA)] = cos_sim(combineAB[pair[0]][2:], combineAB[pair[1]][2:],scaling=scale)*count
    elif LSHType == 'Euclidean':
        for pair, count in pair_count_dict.items():
            if count >= threshold:
                pair_computed += 1
                matching_matrix[pair[0], pair[1]-len(attributesA)] = Euclidean_sim(combineAB[pair[0]][2:], combineAB[pair[1]][2:],scaling=scale)*count
    matching_matrix = matching_matrix.tocsr()
    return matching_matrix, pair_computed

def computeWholeSimMat(attributesA, attributesB, LSHType):
    combineAB = selectAndCombine(attributesA, attributesB)
    combineAB = combineAB.as_matrix()
    sim_vec = []
    scale = np.mean(combineAB[:,2:], axis=0)
    if LSHType == 'Cosine':
        for j in range(len(attributesA)):
            vec = [cos_sim(combineAB[j,2:], combineAB[len(attributesA)+i,2:], scale) for i in range(len(attributesB)) ]
            sim_vec.append(vec)
    elif LSHType == 'Euclidean':
        for j in range(len(attributesA)):
            vec = [Euclidean_sim(combineAB[j,2:], combineAB[len(attributesA)+i,2:], scale) for i in range(len(attributesB)) ]
            sim_vec.append(vec)

    return np.array(sim_vec)


def combineBucketsBySum(buckets, combineAB, Afname):
    pair_count_dict = defaultdict(int)

    for bucket in buckets:
        for buck, collisions in bucket.items(): # collisions = [(Graph, Id)]
            if len(collisions) <= 1:
                continue
            
            A_idx = combineAB[(combineAB['Graph'] == Afname)\
                & (combineAB['Id'].isin([c[1] for c in collisions if c[0]==Afname]))]
            B_idx = combineAB[(combineAB['Graph'] != Afname)\
                & (combineAB['Id'].isin([c[1] for c in collisions if c[0]!=Afname]))]
            
            if len(collisions) == len(A_idx):    # We don't want all in A 
                continue
                
            for aid in A_idx.index.values:
                for bid in B_idx.index.values:
                    pair_count_dict[(combineAB['Id'][aid], combineAB['Id'][bid])] += 1

    return pair_count_dict

def combineBucketsBySumMulti(buckets, stacked_attrs, graphs, center_id):
    pair_count_dict = defaultdict(lambda : defaultdict(int))

    for bucket in buckets:
        for buck, collisions in bucket.items(): # collisions = [(Graph, Id)]
            if len(collisions) <= 1:
                continue
            
            A_idx = stacked_attrs[(stacked_attrs['Graph'] == center_id)\
                & (stacked_attrs['Id'].isin([c[1] for c in collisions if c[0]==center_id]))]

            if len(collisions) == len(A_idx) or len(A_idx) == 0:    # We don't want all in A 
                continue

            for g in graphs:
                if g == center_id:
                    continue
                B_idx = stacked_attrs[(stacked_attrs['Graph'] == g)\
                    & (stacked_attrs['Id'].isin([c[1] for c in collisions if c[0]==g]))]
                for aid in A_idx.index.values:
                    for bid in B_idx.index.values:
                        pair_count_dict[g][(stacked_attrs['Id'][aid], stacked_attrs['Id'][bid])] += 1

    return pair_count_dict


def plotBucketDistribution(bucket):
    plt.bar(range(len(bucket))
            , sorted([len(values) for key, values in bucket.items()])
            , align='center')
    plt.show()


# plot bucket correctness
def plotBucketCorrectness(d, n):
    correct = {}
    for v, k in d.items():
        cnt = 0
        for i in k:
            if (i < n):
                if (i + n in k):
                    cnt += 2
            else:
                break
        correct[v] = cnt
    plt.clf
    plt.figure()
    plt.bar(range(len(d)), [len(v) for k,v in d.items()], alpha=0.5, label='bucket', color='blue')
    plt.bar(range(len(correct)), [correct[k] for k,v in d.items()], alpha=0.5, label='correct', color='green')
    plt.xlabel('bucket')
    plt.ylabel('number')
    plt.legend(loc='best')
    plt.show()

def Rank(matching_matrix, P = None):
    if P is not None:
        matching_matrix = matching_matrix.dot(P)
    n, d = matching_matrix.shape
    
    ranking = np.zeros((n))
    for i in range(n):      
        #rank = n - matching_matrix[i, :].argsort().tolist().index(i)
        if matching_matrix[i,i] != 0:
            rank = sorted(matching_matrix[i, :], reverse = True).index(matching_matrix[i, i]) + 1
            ranking[i] = 1.0 / rank
    return ranking

def sparseRank(matching_matrix, P = None):
    if P:
        matching_matrix = matching_matrix.dot(P)

    n, d = matching_matrix.shape
    ranking = np.zeros((n))
    sorted_row = defaultdict(list)

    matching_matrix = matching_matrix.tocoo() # For .row and .col
    tuples = izip(matching_matrix.row, matching_matrix.col, matching_matrix.data)
    rank = sorted(tuples, key = lambda x: (x[0], x[2]), reverse = True)
    rank = [(pair[0], pair[1]) for pair in rank]
    # Dictionary for each node to other nodes sorted by score
    for r in rank:
        sorted_row[r[0]].append(r[1])
    # Find position of same index
    matching_matrix = matching_matrix.tocsr() # For [i, i]
    for i in range(n):
        if i in sorted_row and matching_matrix[i, i] != 0:
            ranking[i] = 1.0 / (sorted_row[i].index(i) + 1)
    return ranking

def argmaxMatch(matching_matrix, attributesA, attributesB, P = None):
    if P is not None:
        matching_matrix = matching_matrix.dot(P)
    score =[]
    for i in range(matching_matrix.shape[0]):
        score.append(attributesB['Id'][matching_matrix[i].argsort()[-1]] == attributesA['Id'][i])
    return score

def hungarianMatch(matching_matrix, P):
    cost_mat = 100 - matching_matrix
    m = Munkres()
    indexes = m.compute(cost_mat)
    hun_index = [tup[1] for tup in indexes]
    P_index = P.argsort()[:,-1]
    score = [ hun == p for hun, p in zip(hun_index,P_index)]
    return score

