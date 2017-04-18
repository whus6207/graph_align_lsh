from preprocessing_syn import *
from HashAlign import *
from pure_baseline import *

if __name__ == '__main__':
	folders = ['brain']
	euc_widths = {'facebook':[4], 'dblp-A':[0.5]}
	cos_num_planes = {'facebook':[25], 'dblp-A':[40], 'brain':[40]}
	thresholds = [0.10]
	LSHs = ['Cosine']
	band_numbers = {'facebook':[4], 'dblp-A':[2], 'brain':[2]}
	noise_levels = [0.02]
	num_graph = 4
	fname = 'exp_brain'
	for f in folders:
		for noise_level in noise_levels:
			preprocessing(edge_dir = 'Data/'+f+'.edges', save_dir = f, number = num_graph-1, noise_level = noise_level
				, weighted_noise = 1.0, weighted = True)
			final_runner = PureFinal(fname)
			for lsh in LSHs:
				for t in thresholds:
					final_runner.run(f,lsh,1)
			netalign_runner = PureNetAlign(fname)
			for lsh in LSHs:
				for t in thresholds:
					netalign_runner.run(f,lsh,t)

			#preprocessing(edge_dir = 'Data/'+f+'.edges', save_dir = f, number = num_graph-1, noise_level = noise_level)
			hashalign_runner = HashAlign(fname)
			hashalign_runner.run(folders = [f], band_numbers = band_numbers[f], cos_num_plane = cos_num_planes[f]
					, LSHs = LSHs, thresholds = thresholds, compute_netalign = True)
			hashalign_runner.run(folders = [f], band_numbers = band_numbers[f], cos_num_plane = cos_num_planes[f]
					, LSHs = LSHs, thresholds = thresholds, compute_final = True)
