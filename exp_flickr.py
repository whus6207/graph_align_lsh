from HashAlign import *

if __name__ == '__main__':
	folders = ['flickr_myspace']
	cos_num_planes = {'flickr_myspace':[50]}
	thresholds = [0.10]
	LSHs = ['Cosine']
	band_numbers = {'flickr_myspace':[2]}
	fname = 'exp_flickr'
	for f in folders:
		hashalign_runner = HashAlign(fname)
		hashalign_runner.run(folders = [f], band_numbers = band_numbers[f], cos_num_plane = cos_num_planes[f]
				, LSHs = LSHs, thresholds = thresholds, compute_netalign = True)
