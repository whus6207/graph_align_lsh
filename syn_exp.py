import sys
from subprocess import Popen
from HashAlign import *
from pure_netalign import *
from random import shuffle

if __name__ == '__main__':
	testdata = [('facebook', [25], [4]), ('email', [50], [4]), ('dblp', [100], [4])]
	LSHs = ['Cosine', 'Euclidean']
	fname = 'exp_syn'
	noise_level = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

	for data, ce in testdata:
		for noise in noise_level:
			preprocess = 'python2 preprocessing_syn.py ' + 'Data/' + data + '.edges ' + data + ' ' + noise + ' 1'
			Popen(preprocess, shell=True).wait()
			c, e = ce
			for lsh in LSHs:
				# HashAlign
				ha = HashAlign(fname)
				ha.run(cos_num_plane = c, euc_width = e, LSHs = lsh)
				# HashAlign + NetAlign
				ha = HashAlign(fname)
				ha.run(cos_num_plane = c, euc_width = e, compute_netalign = True, LSHs = lsh)
				# HasAlign + Final
				ha = HashAlign(fname)
				ha.run(cos_num_plane = c, euc_width = e, compute_final = True, LSHs = lsh)
				# NetAlign
				na = PureNetalign(fname)
				na.run(filename = data, LSHType = lsh)
				# IsoRank
				# Final
