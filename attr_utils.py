import snap
import numpy as np
import pandas as pd
import random

def getEgoAttr(UGraph, attributes, directed = True):
    egoDeg = np.zeros((UGraph.GetNodes(),))
    egoOutDeg = np.zeros((UGraph.GetNodes(),))
    egoInDeg = np.zeros((UGraph.GetNodes(),))
    egoConn = np.zeros((UGraph.GetNodes(),))
    avgNeighDeg = np.zeros((UGraph.GetNodes(),))
    avgNeighInDeg = np.zeros((UGraph.GetNodes(),))
    avgNeighOutDeg = np.zeros((UGraph.GetNodes(),))

    for NI in UGraph.Nodes():
        thisNID = NI.GetId()
        NIdegree = attributes['Degree'][thisNID]
        if NIdegree == 0:
            print thisNID, 'degree = 0!'
        InNodes = []
        OutNodes = []

        if directed:
            for Id in NI.GetInEdges():
                InNodes.append(Id)

        for Id in NI.GetOutEdges():
            OutNodes.append(Id)
        EgoNodes = set(InNodes+OutNodes+[NI.GetId()])

        egoID = 0
        egoOD = 0
        neighIDsum = 0
        neighODsum = 0
        egoconn = 0
        for Id in InNodes+OutNodes:
            ego_NI = UGraph.GetNI(Id)

            if directed:
                for IID in ego_NI.GetInEdges():
                    neighIDsum += 1
                    if IID not in EgoNodes:
                        egoID += 1
                    else:
                        egoconn += 1

            for OID in ego_NI.GetOutEdges():
                neighODsum += 1
                if OID not in EgoNodes:
                    egoOD += 1
                else:
                    egoconn += 1


        egoDeg[thisNID] = egoID + egoOD
        egoInDeg[thisNID] = egoID
        egoOutDeg[thisNID] = egoOD
        avgNeighDeg[thisNID] = (neighIDsum+neighODsum)/float(NIdegree)
        avgNeighInDeg[thisNID] = neighIDsum/float(NIdegree)
        avgNeighOutDeg[thisNID] = neighODsum/float(NIdegree)
        egoConn[thisNID] = (egoconn+NIdegree)/float(NIdegree+1)

    attributes['EgonetDegree'] = egoDeg
    attributes['AvgNeighborDeg'] = avgNeighDeg
    attributes['EgonetConnectivity'] = egoConn

    if directed:
        attributes['EgonetInDegree'] = egoInDeg
        attributes['EgonetOutDegree'] = egoOutDeg
        attributes['AvgNeighborInDeg'] = avgNeighInDeg
        attributes['AvgNeighborOutDeg'] = avgNeighOutDeg

def getUndirAttribute(filename):
    UGraph = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1)
    # or UGraph.GetNodes()
    attributes = pd.DataFrame(np.zeros(shape=(UGraph.GetNodes(), 10)), 
                              columns=['Graph', 'Id', 'Degree', 'NodeBetweennessCentrality', 
                                       'FarnessCentrality', 'PageRank', 'NodeEccentricity',
                                       'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity'])

    attributes['Graph'] = [filename] * UGraph.GetNodes()#UGraph.GetNodes()
    # Degree
    attributes['Id'] = range(1, UGraph.GetNodes()+1)
    degree = np.zeros((UGraph.GetNodes(),))
    OutDegV = snap.TIntPrV()
    snap.GetNodeOutDegV(UGraph, OutDegV)
    for item in OutDegV:
        degree[item.GetVal1()] = item.GetVal2()
    attributes['Degree'] = degree

    getEgoAttr(UGraph, attributes, directed=False)

    # Farness Centrality, Node Eccentricity
    farCentr = np.zeros((UGraph.GetNodes(),))
    nodeEcc = np.zeros((UGraph.GetNodes(),))
    for NI in UGraph.Nodes():
        farCentr[NI.GetId()] = snap.GetFarnessCentr(UGraph, NI.GetId())
        nodeEcc[NI.GetId()] = snap.GetNodeEcc(UGraph, NI.GetId(), False)
    attributes['FarnessCentrality'] = farCentr
    attributes['NodeEccentricity'] = nodeEcc

    # Betweenness Centrality
    betCentr = np.zeros((UGraph.GetNodes(),))
    Nodes = snap.TIntFltH()
    Edges = snap.TIntPrFltH()
    snap.GetBetweennessCentr(UGraph, Nodes, Edges, 1.0)
    for node in Nodes:
        betCentr[node] = Nodes[node]
    attributes['NodeBetweennessCentrality'] = betCentr

    # PageRank
    pgRank = np.zeros((UGraph.GetNodes(),))
    PRankH = snap.TIntFltH()
    snap.GetPageRank(UGraph, PRankH)
    for item in PRankH:
        pgRank[item] = PRankH[item]
    attributes['PageRank'] = pgRank

    return attributes

def getDirAttribute(filename):
    Graph = snap.LoadEdgeList(snap.PNGraph, filename, 0, 1)
    
    attributeNames = ['Graph', 'Id', 'Degree', 'InDegree', 'OutDegree', 'NodeBetweennessCentrality', 
                      'FarnessCentrality', 'PageRank', 'HubsScore', 'AuthoritiesScore', 'NodeEccentricity',
                      'EgonetDegree', 'EgonetInDegree', 'EgonetOutDegree',
                      'AvgNeighborDeg', 'AvgNeighborInDeg', 'AvgNeighborOutDeg','EgonetConnectivity']

    attributes = pd.DataFrame(np.zeros((Graph.GetNodes(), len(attributeNames))), columns=attributeNames)
    
    attributes['Graph'] = [filename] * Graph.GetNodes()
    attributes['Id'] = range(1, Graph.GetNodes()+1)
    
    # Degree
    degree = np.zeros((Graph.GetNodes(),))
    InDegV = snap.TIntPrV()
    snap.GetNodeInDegV(Graph, InDegV)
    for item in InDegV:
        degree[item.GetVal1()] = item.GetVal2()
    attributes['Degree'] += degree
    attributes['InDegree'] = degree
    
    degree = np.zeros((Graph.GetNodes(),))
    OutDegV = snap.TIntPrV()
    snap.GetNodeOutDegV(Graph, OutDegV)
    for item in OutDegV:
        degree[item.GetVal1()] = item.GetVal2()
    attributes['Degree'] += degree
    attributes['OutDegree'] = degree
    
    getEgoAttr(Graph, attributes)

    attributes['Degree'] /= Graph.GetNodes()
    attributes['InDegree'] /= Graph.GetNodes()
    attributes['OutDegree'] /= Graph.GetNodes()

    # Degree, Closeness, Farness Centrality, Node Eccentricity
    farCentr = np.zeros((Graph.GetNodes(),))
    nodeEcc = np.zeros((Graph.GetNodes(),))
    for NI in Graph.Nodes():
        farCentr[NI.GetId()] = snap.GetFarnessCentr(Graph, NI.GetId(), True, True)
        nodeEcc[NI.GetId()] = snap.GetNodeEcc(Graph, NI.GetId(), True)
    attributes['FarnessCentrality'] = farCentr
    attributes['NodeEccentricity'] = nodeEcc

    # Betweenness Centrality
    betCentr = np.zeros((Graph.GetNodes(),))
    Nodes = snap.TIntFltH()
    Edges = snap.TIntPrFltH()
    snap.GetBetweennessCentr(Graph, Nodes, Edges, 1.0, True)
    for node in Nodes:
        betCentr[node] = Nodes[node]
    attributes['NodeBetweennessCentrality'] = betCentr

    # PageRank
    pgRank = np.zeros((Graph.GetNodes(),))
    PRankH = snap.TIntFltH()
    snap.GetPageRank(Graph, PRankH)
    for item in PRankH:
        pgRank[item] = PRankH[item]
    attributes['PageRank'] = pgRank

    # Hubs, Authorities score 
    hubs = np.zeros((Graph.GetNodes(),))
    auth = np.zeros((Graph.GetNodes(),))
    NIdHubH = snap.TIntFltH()
    NIdAuthH = snap.TIntFltH()
    snap.GetHits(Graph, NIdHubH, NIdAuthH)
    for item in NIdHubH:
        hubs[item] = NIdHubH[item]
    for item in NIdAuthH:
        auth[item] = NIdAuthH[item]
    attributes['HubsScore'] = hubs
    attributes['AuthoritiesScore'] = auth

    return attributes

def addNodeAttribute(structAttributes, nodeAttributeNames = None, nodeAttributeValues = None, P = None):
    if len(nodeAttributeNames) > 0 and len(nodeAttributeValues) > 0:
        if P is not None:
            nodeAttributeValues = P.dot(np.array(nodeAttributeValues))
            nodeAttributes = pd.DataFrame(nodeAttributeValues.astype(int), columns = nodeAttributeNames)
        else:
            nodeAttributes = pd.DataFrame(nodeAttributeValues, columns = nodeAttributeNames)
        structAttributes = pd.concat([structAttributes, nodeAttributes], axis = 1)
    return structAttributes




