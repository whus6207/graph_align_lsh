import sys
from subprocess import Popen
from random import shuffle

if __name__ == '__main__':
	testdata = ['facebook', 'email', 'dblp']
	noise_level = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

	for data in testdata:
		for noise in noise_level:
			preprocess = 'python2 preprocessing_syn.py ' + 'Data/' + data + '.edges ' + data + ' ' + noise + ' 1'
			Popen(preprocess, shell=True).wait()
			hashalign = 'python2 HashAlign.py ' + data 
			Popen(hashalign, shell=True).wait()