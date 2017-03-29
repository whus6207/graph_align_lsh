from scipy.sparse import lil_matrix
from scipy.sparse import identity
import scipy.sparse as sparse
import scipy
def loadSparseGraph(fname, graph_type = 'Undirected'):
    nodes = []
    with open(fname) as f:
        for line in f:
            pair = line.strip().split()
            nodes.append(int(pair[0]))
            nodes.append(int(pair[1]))
    A_size = max(nodes)+1 # Node Id starts from 1
    A = lil_matrix((A_size, A_size))
    for i in range(0,len(nodes),2):
        A[nodes[i], nodes[i+1]]=1
        if graph_type == 'Undirected':
            A[nodes[i+1], nodes[i]]=1
    return A

def removeIsolatedSparse(A):
    A = A.tocsr()
    rest_bool = ((A.sum(axis=0) != 0).tolist() or (A.sum(axis=1) != 0).tolist())[0] # 2d to 1d
    rest_idx = [i for i in xrange(len(rest_bool)) if rest_bool[i]]
    A = A[rest_idx, :]
    A = A[:, rest_idx]
    return A, rest_idx

def permuteSparse(A, is_perm = False, has_noise = False, level = 0.05):
    m, n = A.get_shape()
    perm = scipy.random.permutation(m)

    P = identity(m)
    if is_perm:
        P = P.tocsr()[perm, :]
    B = P.dot(A).dot(P.T)
    # Only flip existinf edges
    if has_noise:
        #sparse.rand(m, n, 0.1).tocsr().ceil().toarray()
        # Flipping existing edges
        noise = [(k, v) for k, v in zip(B.nonzero()[0], B.nonzero()[1]) if k <= v]
        visited = set(noise)  # Remove duplicate edges in undirected
        scipy.random.shuffle(noise)
        noise = noise[0][: int(len(noise[0]) * level)]
        B = B.tolil()
        for pair in noise:
            B[pair[0], pair[1]] = 0
            B[pair[1], pair[0]] = 0
        # Adding edges, should not be visited before
        for _ in range(int(m*m*level)):
            add1, add2 = np.random.choice(m), np.random.choice(m)
            while ((add1, add2) in visited or (add2, add1) in visited):
                add1, add2 = np.random.choice(m), np.random.choice(m)
            B[add1, add2] = 1
            B[add2, add1] = 1
            visited.add((add1, add2))
        B = B.tocsr()

    return B, P

# B is a sparse matrix
def writeSparseToFile(fname, B):
    edges = [zip(B.nonzero()[0], B.nonzero()[1])][0]  # list of tuples
    with open(fname, 'w') as f:
        for e1, e2 in edges:
            f.write(str(e1)+" "+str(e2)+"\n")
    f.close()













