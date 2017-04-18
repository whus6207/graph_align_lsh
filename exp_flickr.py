from HashAlign import *
from pure_baseline import *

if __name__ == '__main__':
	folders = ['flickr_myspace']
	cos_num_planes = {'flickr_myspace':[50, 80]}
	thresholds = [0.20, 0.30]
	LSHs = ['Cosine']
	band_numbers = {'flickr_myspace':[2]}
	fname = 'exp_flickr'
	for f in folders:
		#final_runner = PureFinal(fname)
		#for lsh in LSHs:
		#	final_runner.run(f,lsh,1, all_1 = True)
		hashalign_runner = HashAlign(fname)
		hashalign_runner.run(folders = [f], band_numbers = band_numbers[f], cos_num_plane = cos_num_planes[f]
				, LSHs = LSHs, thresholds = thresholds, compute_final = True)
