import sys
from preprocessing_syn import *
from HashAlign import *
from pure_baseline import *
from random import shuffle

if __name__ == '__main__':
	testdata = [('facebook', [25], [4])]
	b = [4]
	LSHs = ['Cosine']
	fname = 'exp_thre_fb'
	threshold = [1.0, 0.8, 0.5, 0.2, 0.1]

	for data, c, e in testdata:
		preprocessing('Data/' + data + '.edges', 'Data/' + data + '.nodes', data, number = 1, noise_level = 0.05)
		# HashAlign
		ha = HashAlign(fname)
		ha.run(band_numbers = b, cos_num_plane = c, euc_width = e, LSHs = LSHs, folders = [data], thresholds = threshold)
		# HashAlign + NetAlign
		# ha = HashAlign(fname)
		# ha.run(band_numbers = b, cos_num_plane = c, euc_width = e, compute_netalign = True, LSHs = LSHs, folders = [data])
		# HasAlign + Final
		# ha = HashAlign(fname)
		# ha.run(band_numbers = b, cos_num_plane = c, euc_width = e, compute_final = True, LSHs = LSHs, folders = [data])

