import matplotlib.pyplot as plt
import numpy as np 
from attr_utils import *
from io_utils import *
from lsh_utils import *
import pandas as pd
import os.path
import sys
import pickle


def plot(fname, bandNumber, rank_att_score, correct_att_score, pairs_computed):
    plt.figure()
    plt.plot(range(bandNumber, len(rank_att_score)+ bandNumber), rank_att_score, label = 'Ranking', linestyle='--')
    plt.plot(range(bandNumber, len(correct_att_score)+ bandNumber), correct_att_score, label = 'Correct Match', linewidth=2)
    plt.plot(range(bandNumber, len(pairs_computed)+ bandNumber), pairs_computed, label = 'Compute Pairs', linewidth=2)
    plt.legend(loc = 'best')
    plt.xlabel('Numbers of attribute')
    plt.ylabel('Matching score')
    plt.title(fname)
    # plt.title(filename + ", "+ GraphType + ", " + LSHType + ", "+str(bandNumber))
    # fname = "att_fig/"+ GraphType + "_" + LSHType + "_"+str(bandNumber)
    plt.savefig(fname+'.png')

def experiment(df, filename = 'metadata/phys.edges', nodeAttributeFile = None, multipleGraph = False, is_perm = False, 
    has_noise = False, GraphType = 'Directed', bandNumber = 2, adaptiveLSH = False, LSHType = 'Euclidean', 
    loop_num = 3, compute_hungarian = True):
    """
    Experiment on two graphs with multiple setting

    """
    path = 'metadata/' + str(GraphType)

    A = loadGraph(filename, GraphType)
    A, rest_idx = removeIsolatedNodes(A)

    if nodeAttributeFile is not None:
        nodeAttributesValue, nodeAttributesName = loadNodeFeature(nodeAttributeFile)
        #nodeAttributesValue = [nodeAttributesValue[i] for i in rest_idx]
    else:
        nodeAttributesValue, nodeAttributesName = [], []

    if multipleGraph == True:
        pass
    else:
        B, P = permuteNoiseMat(A,is_perm, has_noise)

    # Write edges to file

    writeEdgesToFile(path + '/A.edges', A)
    writeEdgesToFile(path + '/B.edges', B)
    rank_att_score = 0
    rank_score_upper = 0
    correct_att_score = 0
    correct_score_upper = 0
    correct_score_hungarian = 0
    pairs_computed = 0


    for i in range(loop_num):
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
            attributes = ['Degree', 'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']
            attributes = attributes + nodeAttributesName


            if adaptiveLSH == True :
                bandDeg = ['Degree','PageRank','NodeBetweennessCentrality']
                bandEdge = ['EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']
                bandNode = nodeAttributesName[:]

                if LSHType == 'Cosine':
                    bucketDeg = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandDeg), 20)
                    bucketEdge = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandEdge), 20)

                elif LSHType == 'Euclidean':
                    bucketDeg = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandDeg), 2)
                    bucketEdge = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandEdge), 2)


                bucketNode = generateNodeBuckets(LSHType, attributesA, attributesB, bandNode)
                buckets = [bucket for bucket in [bucketDeg, bucketEdge, bucketNode] if len(bucket) > 0]

                for i, bucket in enumerate(buckets):
                    with open(path + '/buckets-band-' + str(i+1), 'w') as f:
                        for k, v in bucket.items():
                            f.write(str(k) + str(v) + '\n')
                
     
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
            attributes = ['Degree', 'EgonetDegree', 'EgonetInDegree', 'EgonetOutDegree',
                          'AvgNeighborDeg', 'AvgNeighborInDeg', 'AvgNeighborOutDeg','EgonetConnectivity']
            attributes = attributes + nodeAttributesName

            if adaptiveLSH == True:
                bandDeg = ['Degree','InDegree','OutDegree']
                bandEgo = ['EgonetDegree', 'EgonetInDegree', 'EgonetOutDegree',
                          'AvgNeighborDeg', 'AvgNeighborInDeg', 'AvgNeighborOutDeg','EgonetConnectivity']
                bandCentr = ['PageRank', 'NodeBetweennessCentrality', 
                             'HubsScore', 'AuthoritiesScore']
                bandNode = nodeAttributesName[:]

                if LSHType == 'Cosine':

                    bucketDeg = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandDeg), 20)
                    bucketEgo = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandEgo), 20)
                    bucketCentr = generateCosineBuckets(selectAndCombine(attributesA, attributesB, bandCentr), 20)

                elif LSHType == 'Euclidean':
                    bucketDeg = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandDeg), 2)
                    bucketEgo = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandEgo), 2)
                    bucketCentr = generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, bandCentr), 2)

                bucketNode = generateNodeBuckets(LSHType, attributesA, attributesB, bandNode)
                buckets = [bucket for bucket in [bucketDeg, bucketEgo, bucketCentr, bucketNode] if len(bucket) > 0]

                for i, bucket in enumerate(buckets):
                    with open(path+'/buckets-band'+str(i+1), 'w') as f:
                        for k, v in bucket.items():
                            f.write(str(k) + str(v) + '\n')

            """
            Compute acuuracy to number of attributes
            """

        

        combineAB = selectAndCombine(attributesA, attributesB)
        sim_matrix = computeWholeSimMat(attributesA, attributesB, LSHType)

        Best_Ranking = Rank(sim_matrix, P)
        Best_correctMatch = argmaxMatch(sim_matrix, attributesA, attributesB, P)
        if compute_hungarian:
            hung_score = hungarianMatch(sim_matrix, P)
        for _ in range(5):
            randomBand = []
            rank_acc_att =[]
            correct_acc_att = []
            pairs_count_att = []
            np.random.shuffle(attributes)
            for index in range(1, len(list(attributes)) + 1):
                sub_attributes = list(attributes[: index])
                #np.random.shuffle(sub_attributes)
                if len(sub_attributes) >= bandNumber:

                    print "- - - - - - - - - - - - - - - - -"
                    print "Attributes include: ",
                    sys.stdout.write(sub_attributes[0])
                    for item in sub_attributes[1:]:
                        print ", " + item,
                    print ""  

                    randomBand = [sub_attributes[i*len(sub_attributes)/bandNumber: (i + 1)*len(sub_attributes)/bandNumber]for i in range(bandNumber)]
                    
                    buckets = []

                    if LSHType == 'Cosine':
                        for band in randomBand:
                            buckets.append(generateCosineBuckets(selectAndCombine(attributesA, attributesB, band), 20))


                    elif LSHType == 'Euclidean':
                        for band in randomBand:
                            buckets.append(generateEuclideanBuckets(selectAndCombine(attributesA, attributesB, band), 2))
                    #bucket_all.append(buckets)

                    pair_count_dict = combineBucketsBySum(buckets, combineAB, path+'/A.edges')
                    matching_matrix, this_pair_computed = computeMatchingMat(attributesA, attributesB, pair_count_dict, LSHType, threshold=1)
                    pairs_count_att.append(this_pair_computed/float(matching_matrix.shape[0]*matching_matrix.shape[1]))      

                    Ranking = Rank(matching_matrix, P)                    
                    ranking_match_score = sum(Ranking)/float(len(Ranking))
                    rank_acc_att.append(ranking_match_score)

                    correctMatch = argmaxMatch(matching_matrix, attributesA, attributesB, P)                   
                    correct_match_score = sum(correctMatch) / float(len(correctMatch))
                    correct_acc_att.append(correct_match_score)

                    

                    #print "matching score by ranking: %f" %ranking_match_score,
                    ##print ", matching score by correct match: %f" %correct_match_score
        
            rank_att_score += np.array(rank_acc_att)
            correct_att_score += np.array(correct_acc_att)
            pairs_computed += np.array(pairs_count_att)
            

        rank_score_upper += sum(Best_Ranking)/float(len(Best_Ranking))
        correct_score_upper += sum(Best_correctMatch) / float(len(Best_correctMatch))
        if compute_hungarian:
            correct_score_hungarian += sum(hung_score)/float(len(hung_score))
        else:
            correct_score_hungarian += 0

        #rank_acc_att += rank_acc_att
        #correct_acc_att += correct_acc_att

    rank_att_score = rank_att_score / (float(loop_num) * 5)
    rank_score_upper /= loop_num
    correct_att_score = correct_att_score / (float(loop_num) * 5)
    correct_score_upper /= loop_num
    correct_score_hungarian /= loop_num
    pairs_computed = pairs_computed / (float(loop_num) * 5)


    print "=========================================================="
    print "nodeAttributeFile = " + str(nodeAttributeFile)
    print "is_perm = " + str(is_perm) + ", has_noise = "+ str(has_noise)+", GraphType = "+ GraphType
    print "bandNumber = "+str(bandNumber)+", adaptiveLSH = "+ str(adaptiveLSH)+", LSHType = "+LSHType

    #print "matching score by ranking upper bound: %f" %(sum(Best_ranking)/len(Best_ranking))
    #score = hungarianMatch(sim_matrix, P)
    #upper_bound_score = sum(score)/float(len(score))
    #print "hungarian matching score upper bound: %f" %(upper_bound_score)
    #print "matching score by correct match: %f" % (sum(correctMatch) / float(len(correctMatch)))
    #print "matching score by correct match upper bound %f" % (sum(Best_correctMatch) / float(len(Best_correctMatch)))
    #print "percentage of pairs computed: %f" %(len(pair_count_dict)/float(matching_matrix.shape[0]*matching_matrix.shape[1]))

    for i in range(len(rank_att_score)):   
        df = df.append({'filename':filename, 'nodeAttributeFile': str(nodeAttributeFile),'is_perm':is_perm\
            , 'has_noise':has_noise, 'GraphType':GraphType\
            , 'bandNumber':bandNumber, 'adaptiveLSH':adaptiveLSH, 'LSHType':LSHType\
            , 'rank_score' : rank_att_score[i]\
            , 'rank_score_upper' : rank_score_upper\
            , 'correct_score' : correct_att_score[i]\
            , 'correct_score_upper' : correct_score_upper\
            , 'correct_score_hungarian' : correct_score_hungarian\
            , 'pairs_computed' : pairs_computed[i]\
            }, ignore_index=True)

    

    return df


if __name__ == '__main__':
    adaptiveLSH = [False]
    noise = [True]
    bandNumber = [2,4,8]
    fname = 'exp_result_att.pkl'

    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            df = pickle.load(f)
    else:
        df = pd.DataFrame(
            columns=['filename', 'nodeAttributeFile', 'is_perm', 'has_noise', 'GraphType', 'bandNumber', 'adaptiveLSH', 'LSHType'\
                , 'rank_score', 'correct_score'\
                , 'pairs_computed'])

    for a in adaptiveLSH:
        for n in noise:
            for b in bandNumber:
                df = experiment(df, filename = 'facebook/0.edges', nodeAttributeFile = 'fb.nodes', 
                    multipleGraph = False, is_perm = False, has_noise = n, GraphType = 'Undirected', 
                    bandNumber = b, adaptiveLSH = a, LSHType = 'Cosine')
                df = experiment(df, filename = 'facebook/0.edges', nodeAttributeFile = 'fb.nodes', 
                    multipleGraph = False, is_perm = False, has_noise = n, GraphType = 'Undirected', 
                    bandNumber = b, adaptiveLSH = a, LSHType = 'Euclidean')
                df = experiment(df, filename = 'metadata/phys.edges', nodeAttributeFile = 'phys.nodes', 
                    multipleGraph = False, is_perm = False, has_noise = n, GraphType = 'Directed', 
                    bandNumber = b, adaptiveLSH = a, LSHType = 'Cosine')
                df = experiment(df, filename = 'metadata/phys.edges', nodeAttributeFile = 'phys.nodes', 
                    multipleGraph = False, is_perm = False, has_noise = n, GraphType = 'Directed', 
                    bandNumber = b, adaptiveLSH = a, LSHType = 'Euclidean')

                df = experiment(df, filename = 'email/email.edges', nodeAttributeFile = 'email.nodes', 
                    multipleGraph = False, is_perm = False, has_noise = n, GraphType = 'Undirected', 
                    bandNumber = b, adaptiveLSH = a, LSHType = 'Cosine', compute_hungarian = False)
                df = experiment(df, filename = 'email/email.edges', nodeAttributeFile = 'email.nodes', 
                    multipleGraph = False, is_perm = False, has_noise = n, GraphType = 'Undirected', 
                    bandNumber = b, adaptiveLSH = a, LSHType = 'Euclidean', compute_hungarian = False)

                if a:
                    break

    """
    df = experiment(df, filename = 'facebook/0.edges', nodeAttributeFile = 'fb.nodes', multipleGraph = False, is_perm = False, 
                  has_noise = True, GraphType = 'Undirected', bandNumber = 2, adaptiveLSH = False, LSHType = 'Cosine')
    """
    pickle.dump(df, open(fname,'wb'))


    writer = pd.ExcelWriter('exp_result_attr.xlsx')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
     
    # plot cosine and euc 

    bandNumber = [2,4,8]
    Graph = ['facebook/0.edges', 'metadata/phys.edges', 'email/email.edges']
    # LSHType = ['Cosine', 'Euclidean']
    for b in bandNumber:
        for g in Graph:
            cos_score = df.ix[(df['filename'] == g) & (df['bandNumber'] == b) & (df['LSHType'] == 'Cosine')]
            rank_att_score = cos_score['rank_score']
            correct_att_score = cos_score['correct_score']
            pairs_computed = cos_score['pairs_computed']
            plt.figure()
            plt.plot(range(b, len(rank_att_score)+ b), rank_att_score, label = 'rank_score(cos)', linestyle='--')
            plt.plot(range(b, len(correct_att_score)+ b), correct_att_score, label = 'correct_score(cos)', linewidth=2)
            plt.plot(range(b, len(pairs_computed)+ b), pairs_computed, label = '% pairs computed(cos)', linewidth=2)
            euc_score = df.ix[(df['filename'] == g) & (df['bandNumber'] == b) & (df['LSHType'] == 'Euclidean')]
            rank_att_score = euc_score['rank_score']
            correct_att_score = euc_score['correct_score']
            pairs_computed = euc_score['pairs_computed']
            plt.plot(range(b, len(rank_att_score)+ b), rank_att_score, label = 'rank_score(euc)', linestyle='--')
            plt.plot(range(b, len(correct_att_score)+ b), correct_att_score, label = 'correct_score(euc)', linewidth=2)
            plt.plot(range(b, len(pairs_computed)+ b), pairs_computed, label = '% pairs computed(euc)', linewidth=2)
            plt.legend(loc = 'best')
            plt.xlabel('Numbers of attribute')
            # plt.ylabel('Matching score')
            plt.title(g + ", "+str(b))
            fname = "./att_fig/att/"+ g[0] + "_" + str(b)
            plt.savefig(fname+'.png')






