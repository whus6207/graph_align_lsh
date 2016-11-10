import numpy as np

def loadGraph(fname, graph_type='u'):
    nodes = []
    with open(fname) as f:
        for line in f:
            pair = line.strip().split()
            nodes.append(int(pair[0]))
            nodes.append(int(pair[1]))

    A_size = max(nodes)+1
    A = np.zeros((A_size,A_size))

    for i in range(0,len(nodes),2):
        A[nodes[i]][nodes[i+1]]=1
        if graph_type == 'u':
            A[nodes[i+1]][nodes[i]]=1
    
    return A

def permuteNoiseMat(A, is_perm=False, has_noise=False):
    perm = np.random.permutation(len(A))

    P = np.identity(len(A))
    if is_perm:
        P = P[perm]

    B = P.dot(A).dot(P.T)
    if has_noise:
        noise = np.random.choice([0, 1], size=(len(A),len(A)), p=[99.95/100, 0.05/100])
        B = (B + noise + noise.T)%2
    
    return B

def writeEdgesToFile(fname, B):
    with open(fname, 'w') as f:
        for i in range(len(B)):
            for j in range(len(B)):
                if B[i][j]>0:
                    f.write(str(i)+" "+str(j)+"\n")
