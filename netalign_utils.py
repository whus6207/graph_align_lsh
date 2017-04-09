import matlab.engine as mateng
import matlab
import numpy as np
from scipy  import io

def getNetalignScore(A, B, L):
  eng = mateng.start_matlab()

  # eng.saveMat(matlab.double(A.tolist()), matlab.double(B.tolist()), matlab.double(L.tolist()), s, nargout=0)
  io.savemat('temp.mat', dict(A=A, B=B, L=L))
  accuracy = eng.runNetalign(nargout=1)
  print "netalign: " + str(accuracy)

  eng.quit()

  return accuracy