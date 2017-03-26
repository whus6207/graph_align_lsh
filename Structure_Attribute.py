import snap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def getAttribute(filename):
    UGraph = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1)
    UGraph.Dump()

    attributes = pd.DataFrame(np.zeros(shape=(UGraph.GetNodes(), 12)), 
                              columns=['Graph', 'Id', 'Degree', 'DegreeCentrality', 'NodeBetweennessCentrality', 
                                       'ClosenessCentrality', 'FarnessCentrality', 'PageRank', 'HubsScore', 
                                       'AuthoritiesScore', 'NodeEccentricity', 'EigenvectorCentrality'])
    
    attributes['Graph'] = [filename] * UGraph.GetNodes()
    
    # Degree
    id = []
    degree = []
    OutDegV = snap.TIntPrV()
    snap.GetNodeOutDegV(UGraph, OutDegV)
    for item in OutDegV:
        id.append(item.GetVal1())
        degree.append(item.GetVal2())
    attributes['Id'] = id
    attributes['Degree'] = degree

    # Degree, Closeness, Farness Centrality, Node Eccentricity
    degCentr = []
    cloCentr = []
    farCentr = []
    nodeEcc = []
    for NI in UGraph.Nodes():
        degCentr.append(snap.GetDegreeCentr(UGraph, NI.GetId()))
        cloCentr.append(snap.GetClosenessCentr(UGraph, NI.GetId()))
        farCentr.append(snap.GetFarnessCentr(UGraph, NI.GetId()))
        nodeEcc.append(snap.GetNodeEcc(UGraph, NI.GetId(), False))
    attributes['DegreeCentrality'] = degCentr
    attributes['ClosenessCentrality'] = cloCentr
    attributes['FarnessCentrality'] = farCentr
    attributes['NodeEccentricity'] = nodeEcc

    # Betweenness Centrality
    betCentr = []
    Nodes = snap.TIntFltH()
    Edges = snap.TIntPrFltH()
    snap.GetBetweennessCentr(UGraph, Nodes, Edges, 1.0)
    for node in Nodes:
        betCentr.append(Nodes[node])
    attributes['NodeBetweennessCentrality'] = betCentr

    # PageRank
    pgRank = []
    PRankH = snap.TIntFltH()
    snap.GetPageRank(UGraph, PRankH)
    for item in PRankH:
        pgRank.append(PRankH[item])
    attributes['PageRank'] = pgRank

    # Hubs, Authorities score 
    hubs = []
    auth = []
    NIdHubH = snap.TIntFltH()
    NIdAuthH = snap.TIntFltH()
    snap.GetHits(UGraph, NIdHubH, NIdAuthH)
    for item in NIdHubH:
        hubs.append(NIdHubH[item])
    for item in NIdAuthH:
        auth.append(NIdAuthH[item])
    attributes['HubsScore'] = hubs
    attributes['AuthoritiesScore'] = auth

    # Eigenvector Centrality
    eigenCentr = []
    NIdEigenH = snap.TIntFltH()
    snap.GetEigenVectorCentr(UGraph, NIdEigenH)
    for item in NIdEigenH:
        eigenCentr.append(NIdEigenH[item])
    attributes['EigenvectorCentrality'] = eigenCentr

    return attributes

attributesA = getAttribute("0.edges")
#with open('metadata/attributesA', 'w') as f:
#    for index, row in attributesA.iterrows():
#        f.write(str(attributesA.ix[index]))
#print attributesA

attributesB = getAttribute("B.edges")
#with open('metadata/attributesB', 'w') as f:
#    for index, row in attributesB.iterrows():
#        f.write(str(attributesB.ix[index]))
#print attributesB

# ====================================================

attributes_name = list(attributesA.columns[2:])
attributes_list = []

for name in attributes_name:
    d_list = list(dict(Counter(attributesA[name])).items())
    d_list= sorted(d_list, reverse = True)
    attributes_list.append(d_list)
    


y = []
x =[]
for i in range(len(attributes_list)):
    y.append([ d[0] for d in attributes_list[i]])
    x.append([ d[1] for d in attributes_list[i]])



f1, ax1 = plt.subplots(5, 2, figsize=(10,8), )


for i in range(5):

    ax1[i, 0].plot(y[i], x[i], '.')
    ax1[i, 0].set_title(attributes_name[i])
    ax1[i, 1].set_title('Transformation')
    if i == 2:
        ax1[i, 1].hist(list(np.log(np.sqrt(attributesA[attributes_name[i]] + 1.5))))
        continue
    if i == 3:
        max_value = max(attributesA[attributes_name[i]]) + 1
        ax1[i, 1].hist(list(np.sqrt(max_value - attributesA[attributes_name[i]])), bins = 60)
        continue
    if i == 4:
        ax1[i, 1].hist(np.log(np.log((attributesA[attributes_name[i]]))), bins = 80)
        continue
    ax1[i, 1].hist(list(np.log(np.sqrt(attributesA[attributes_name[i]] + 0.001))))



plt.tight_layout()

f2, ax2 = plt.subplots(5, 2, figsize=(10, 8), )

for j in range(5, 10):

    ax2[j - 5, 0].plot(y[j], x[j], '.')
    ax2[j - 5, 0].set_title(attributes_name[j])
    ax2[j - 5, 1].set_title('Transformation')
    if j == 6:
 
        max_value = max(attributesA[attributes_name[j]] ** (1.0 / 4)) + 1
        ax2[j - 5, 1].hist(list(np.log(max_value - (attributesA[attributes_name[j]]) ** (1.0 / 4))), bins = 100)
        continue
    if j == 7:
        max_value = max(attributesA[attributes_name[j]]) + 1
        ax2[j - 5, 1].hist(list((attributesA[attributes_name[j]]) ** (1.0 / 4)), bins = 60)
        continue
    if j == 8:

        ax2[j - 5, 1].hist(list(attributesA[attributes_name[j]]), bins = 60)
        continue
    if j == 9:
        min_value = 1 - min(attributesA[attributes_name[j]])

        ax2[j - 5, 1].hist(list(attributesA[attributes_name[j]] ** (1.0 / 4)), bins = 60)
        continue

    ax2[j - 5, 1].hist(list(np.log(np.sqrt(attributesA[attributes_name[j]]))))

plt.tight_layout()




plt.show()












