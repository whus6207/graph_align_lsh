import snap
import numpy as np
import pandas as pd

def getUndirAttribute(filename):
    UGraph = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1)
    UGraph.Dump()

    attributes = pd.DataFrame(np.zeros(shape=(UGraph.GetNodes(), 7)), 
                              columns=['Graph', 'Id', 'Degree', 'NodeBetweennessCentrality', 
                                       'FarnessCentrality', 'PageRank', 'NodeEccentricity'])

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

    # Farness Centrality, Node Eccentricity
    farCentr = []
    nodeEcc = []
    for NI in UGraph.Nodes():
        farCentr.append(snap.GetFarnessCentr(UGraph, NI.GetId()))
        nodeEcc.append(snap.GetNodeEcc(UGraph, NI.GetId(), False))
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

    return attributes

def getDirAttribute(filename):
    Graph = snap.LoadEdgeList(snap.PNGraph, filename, 0, 1)
    
    attributeNames = ['Graph', 'Id', 'Degree', 'InDegree', 'OutDegree', 'NodeBetweennessCentrality', 
                      'FarnessCentrality', 'PageRank', 'HubsScore', 'AuthoritiesScore', 'NodeEccentricity']

    attributes = pd.DataFrame(np.zeros((Graph.GetNodes(), len(attributeNames))), columns=attributeNames)
    
    attributes['Graph'] = [filename] * Graph.GetNodes()
    attributes['Id'] = range(1, Graph.GetNodes()+1)
    
    # Degree
    degree = np.zeros((Graph.GetNodes(),))
    InDegV = snap.TIntPrV()
    snap.GetNodeInDegV(Graph, InDegV)
    for item in InDegV:
        degree[item.GetVal1()-1] = item.GetVal2()
    attributes['Degree'] += degree
    attributes['InDegree'] = degree
    
    degree = np.zeros((Graph.GetNodes(),))
    OutDegV = snap.TIntPrV()
    snap.GetNodeOutDegV(Graph, OutDegV)
    for item in OutDegV:
        degree[item.GetVal1()-1] = item.GetVal2()
    attributes['Degree'] += degree
    attributes['OutDegree'] = degree
    
    attributes['Degree'] /= Graph.GetNodes()
    attributes['InDegree'] /= Graph.GetNodes()
    attributes['OutDegree'] /= Graph.GetNodes()

    # Degree, Closeness, Farness Centrality, Node Eccentricity
    farCentr = np.zeros((Graph.GetNodes(),))
    nodeEcc = np.zeros((Graph.GetNodes(),))
    for NI in Graph.Nodes():
        farCentr[NI.GetId()-1] = snap.GetFarnessCentr(Graph, NI.GetId(), True, True)
        nodeEcc[NI.GetId()-1] = snap.GetNodeEcc(Graph, NI.GetId(), True)
    attributes['FarnessCentrality'] = farCentr
    attributes['NodeEccentricity'] = nodeEcc

    # Betweenness Centrality
    betCentr = np.zeros((Graph.GetNodes(),))
    Nodes = snap.TIntFltH()
    Edges = snap.TIntPrFltH()
    snap.GetBetweennessCentr(Graph, Nodes, Edges, 1.0, True)
    for node in Nodes:
        betCentr[node-1] = Nodes[node]
    attributes['NodeBetweennessCentrality'] = betCentr

    # PageRank
    pgRank = np.zeros((Graph.GetNodes(),))
    PRankH = snap.TIntFltH()
    snap.GetPageRank(Graph, PRankH)
    for item in PRankH:
        pgRank[item-1] = PRankH[item]
    attributes['PageRank'] = pgRank

    # Hubs, Authorities score 
    hubs = np.zeros((Graph.GetNodes(),))
    auth = np.zeros((Graph.GetNodes(),))
    NIdHubH = snap.TIntFltH()
    NIdAuthH = snap.TIntFltH()
    snap.GetHits(Graph, NIdHubH, NIdAuthH)
    for item in NIdHubH:
        hubs[item-1] = NIdHubH[item]
    for item in NIdAuthH:
        auth[item-1] = NIdAuthH[item]
    attributes['HubsScore'] = hubs
    attributes['AuthoritiesScore'] = auth

    return attributes