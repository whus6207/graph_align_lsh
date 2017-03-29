from io_utils import *
import random 

number = 5


#A = loadGraph('facebook/0.edges', 'Undirected')
A = loadGraph('./email/email.edges', 'Undirected')
#A = loadGraph('./metadata/phys.edges', 'Directed')
A, rest_idx = removeIsolatedNodes(A)
n = len(A)
attributeNames = ['a' + str(i) for i in range(number)]
attributeValues = [[random.randint(0, j) for i in range(n)] for j in range(1, number * 4 + 1, 4)]
attributes = [attributeNames] + zip(*attributeValues)
#with open('fb.nodes', 'w') as f:
with open('email.nodes', 'w') as f:
#with open('phys.nodes', 'w') as f:
	for line in attributes:
		f.write(" ".join([str(elements) for elements in line]) + "\n")

