import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random

def generateCosineBuckets(attributes, cols):
    randMatrix = np.random.random((attributes.shape[1], cols))
    #randMatrix = np.multiply(((randMatrix - 0.5) * 2).T, 1 / np.median(attributes, axis=0)).T
    randMatrix = np.multiply(((randMatrix - 0.5) * 2).T, 1 / np.mean(attributes, axis=0)).T
    #randMatrix = (randMatrix - 0.5) * 2
    signMatrix = attributes.dot(randMatrix)
    signMatrix[signMatrix > 0] = 1
    signMatrix[signMatrix < 0] = 0
    hashMatrix = signMatrix.dot([2**i for i in range(cols)])
    dic = defaultdict(list)
    for i in range(len(hashMatrix)):
        dic[hashMatrix[i]].append(i)
    return dic

def generateEuclideanBuckets(attributes, bin_wid):
    randVec = np.random.normal(size=(attributes.shape[1],))
    randVec = np.multiply((randVec).T, 10 / np.mean(attributes, axis=0)).T
    bias = random.uniform(0, bin_wid)
    
    hashVec = (attributes.dot(randVec)+bias)/bin_wid
    hashVec = hashVec.astype(int)
    dic = defaultdict(list)
    for i in range((len(hashVec))):
        dic[hashVec[i]].append(i)
    return dic

def selectAndCombine(A, B, cols = None):
    if cols is not None:
        return np.vstack((A[cols].as_matrix(), B[cols].as_matrix()))
    else:
        return np.vstack((A.as_matrix(), B.as_matrix())) 

def cos_sim(v1, v2, scaling=None):
    if scaling is None:
        scaling = np.ones((len(v1),))
    v1 = np.multiply(v1, 1/scaling)
    v2 = np.multiply(v2, 1/scaling)
    return v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def KL_sim(v1, v2):
    v1 = v1.astype(float) / np.mean(v1)
    v2 = v2.astype(float) / np.mean(v2)
    temp = np.log(v1 / v2)
    temp[np.isinf(temp)] = 0
    return np.sum(temp * v1, axis = 0)


def computeMatchingMat(attributesA, attributesB, pair_count_dict):
    combineAB = selectAndCombine(attributesA, attributesB)
    matching_matrix = np.zeros((len(attributesA), len(attributesB)))
    scale = np.mean(combineAB[:,2:], axis=0)
                
    for pair, count in pair_count_dict.items():
        matching_matrix[pair[0]][pair[1]-len(attributesA)] = cos_sim(combineAB[pair[0]][2:], combineAB[pair[1]][2:],scaling=scale)*count
        
    return matching_matrix

def computeWholeSimMat(attributesA, attributesB):
    combineAB = selectAndCombine(attributesA, attributesB)
    sim_vec = []
    scale = np.mean(combineAB[:,2:], axis=0)
    for j in range(len(attributesA)):
        vec = [cos_sim(combineAB[j,2:], combineAB[len(attributesA)+i,2:], scale) for i in range(len(attributesB)) ]
        sim_vec.append(vec)

    return np.array(sim_vec)

def combineBucketsBySum(buckets, combineAB, Afname):
    pair_count_dict = defaultdict(int)

    for bucket in buckets:
        for buck, collisions in bucket.items():
            if len(collisions) <= 1:
                continue
            A_idx = combineAB[collisions,0] == Afname
            B_idx = ~A_idx
            colli_arr = np.array(collisions)
            
            if sum(A_idx) == len(A_idx):    # Not all in A 
                continue
                
            for aid in colli_arr[A_idx]:
                for bid in colli_arr[B_idx]:
                    pair_count_dict[(aid, bid)] += 1

    return pair_count_dict


def plotBucketDistribution(bucket):
    plt.bar(range(len(bucket))
            , sorted([len(values) for key, values in bucket.items()])
            , align='center')
    plt.show()


def Rank(matching_matrix, P = None):
    if P is not None:
        matching_matrix = matching_matrix.dot(P)
    n, d = matching_matrix.shape
    
    ranking = np.zeros((n))
    for i in range(n):      
        #rank = n - matching_matrix[i, :].argsort().tolist().index(i)
        rank = sorted(matching_matrix[i, :], reverse = True).index(matching_matrix[i, i]) + 1
        ranking[i] = 1.0 / rank
    return ranking

