import numpy as np

def computeEuclideanSimilarity(a,b):
  assert (len(a) == len(b)), "Dimension is different"
  eucDis = sum((a - b) ** 2) ** 0.5
  return 1/(1+eucDis)

def computeMatrixEuclidean(a,b):
  retMat = np.zeros((len(a),len(b)))
  for i in range(len(a)):
    for j in range(len(b)):
      retMat[i,j] = computeEuclideanSimilarity(a[i],b[j])
  return retMat

if __name__ == '__main__':
  a = np.random.rand(3,3)
  b = np.random.rand(3,3)
  print (a, b, computeMatrixEuclidean(a,b))
