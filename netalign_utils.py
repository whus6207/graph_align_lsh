import matlab.engine as mateng
import matlab
import numpy as np
from scipy  import io

def getNetalignScore(A, B, L, Pa, Pb):
  eng = mateng.start_matlab()
  # eng.saveMat(matlab.double(A.tolist()), matlab.double(B.tolist()), matlab.double(L.tolist()), s, nargout=0)
  io.savemat('temp.mat', dict(A=A, B=B, L=L, Pa=Pa, Pb=Pb))
  accuracy = eng.runNetalign(nargout=1)
  print "netalign: " + str(accuracy)

  eng.quit()

  return accuracy

def getFinalScore(A, B, H, Pa, Pb):
	eng = mateng.start_matlab()

	io.savemat('temp_final.mat', dict(A = A, B = B, H = H, Pa = Pa, Pb = Pb))

	accuracy = eng.runFinal()
	prtin "final: " + str(accuracy)

	eng.quit()

	return accuracy