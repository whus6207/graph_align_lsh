import sys
from preprocessing_syn import *
from HashAlign import *
# from pure_baseline import *
from random import shuffle

if __name__ == '__main__':
	testdata = [('brain_graphs', [40], [4])]
	b = [4]
	LSHs = ['Cosine']
	fname = 'exp_brain'

	for data, c, e in testdata:
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
			fn.run(filename = data, LSHType = lsh)

