import matlab.engine as mateng
import matlab
import numpy as np
import scipy.io

def getNetalignScore(A, B, sim_mat, s):
  #print (A.shape, B.shape, sim_mat.shape)
  eng = mateng.start_matlab()
  #sim_mat = matlab.double(sim_mat.tolist())
  #scipy.io.savemat('metadata/netalign.mat', dict(A=A, B=B, L=sim_mat))

  # ma, mb, mi, overlap, weight = eng.executeNetalign(matlab.double(A.tolist()), matlab.double(B.tolist()), matlab.double(sim_mat.tolist()), nargout=5)

  # ma = np.array(ma._data).astype(int)
  # mb = np.array(mb._data).astype(int)
  eng.saveMat(matlab.double(A.tolist()), matlab.double(B.tolist()), matlab.double(sim_mat.tolist()), s, nargout=0)

  eng.quit()

  # return float(sum(ma == mb)) / len(ma)
  return 0
