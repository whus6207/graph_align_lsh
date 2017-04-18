import sys
from preprocessing_syn import *
from HashAlign import *
from pure_baseline import *
from random import shuffle

if __name__ == '__main__':
	# testdata = [('facebook', [25], [4]), ('email', [50], [4]), ('dblp-A', [100], [4])]
	testdata = [('email', [50], [4])]
	b = [4]
	LSHs = ['Cosine']
	fname = 'test'
	noise_level = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

	for data, c, e in testdata:
		if data == 'dblp-A':
			b = [2]
			noise_level = [0.1, 0.2, 0.3, 0.5]
		for noise in noise_level:
			preprocessing('Data/' + data + '.edges', 'Data/' + data + '.nodes', data, number = 1, noise_level = noise, node_label = True)
			# HashAlign
			ha = HashAlign(fname)
			ha.run(band_numbers = b, cos_num_plane = c, euc_width = e, LSHs = LSHs, folders = [data])
			# HashAlign + NetAlign
			ha = HashAlign(fname)
			ha.run(band_numbers = b, cos_num_plane = c, euc_width = e, compute_netalign = True, LSHs = LSHs, folders = [data])
			# HasAlign + Final
			ha = HashAlign(fname)
			ha.run(band_numbers = b, cos_num_plane = c, euc_width = e, compute_final = True, LSHs = LSHs, folders = [data])
			for lsh in LSHs:
				# NetAlign
				na = PureNetAlign(fname)
				na.run(filename = data, LSHType = lsh)
				# IsoRank
				ir = PureIsoRank(fname)
				ir.run(filename = data, LSHType = lsh)
				# Final
				fn = PureFinal(fname)
				fn.run(filename = data, LSHType = lsh, threshold = 1.0)

