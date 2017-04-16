import matlab.engine as mateng
import matlab
import numpy as np
from scipy  import io
from scipy.sparse import csr_matrix

def getCSRMatrix(A, ma, mb):
	accuracy, ma, mb = eng.runNetalign(nargout=3)
	row = np.array([x-1 for x in map(list, zip(*ma))[0]])
	col = np.array([x-1 for x in map(list, zip(*mb))[0]])
	data = len(row)*[1]
	return csr_matrix((data, (row, col)), shape=A.shape)

def getNetalignScore(A, B, L, Pa, Pb):
	eng = mateng.start_matlab()
	# eng.saveMat(matlab.double(A.tolist()), matlab.double(B.tolist()), matlab.double(L.tolist()), s, nargout=0)
	io.savemat('temp.mat', dict(A=A, B=B, L=L, Pa=Pa, Pb=Pb))
	accuracy, ma, mb = eng.runNetalign(nargout=3)
	matching_matrix = getCSRMatrix(A, ma, mb)
	print "netalign: " + str(accuracy)

	eng.quit()

	return accuracy, matching_matrix

def getFinalScore(A, B, H, Pa, Pb, node_A = None, node_B = None):
	eng = mateng.start_matlab()

	if not node_A and not node_B:	
		node_A = np.ones((A.get_shape()[0], 1));
		node_B = np.ones((B.get_shape()[0], 1));
	io.savemat('temp_final.mat', dict(A = A, B = B, H = H, Pa = Pa, Pb = Pb, node_A = node_A, node_B = node_B))

	accuracy, ma, mb = eng.runFinal(nargout=1)
	matching_matrix = getCSRMatrix(A, ma, mb)

	print "final: " + str(accuracy)

	eng.quit()

	return accuracy, matching_matrix

def getIsoRankScore(A, B, L, Pa, Pb):
	eng = mateng.start_matlab()
	# eng.saveMat(matlab.double(A.tolist()), matlab.double(B.tolist()), matlab.double(L.tolist()), s, nargout=0)
	io.savemat('temp_iso.mat', dict(A=A, B=B, L=L, Pa=Pa, Pb=Pb))
	
	accuracy = eng.runIsoRank(nargout=1)

	print "isorank: " + str(accuracy)

	eng.quit()

	return accuracy