from preprocessing_syn import *
from HashAlign import *
from pure_baseline import *

if __name__ == '__main__':
	folders = ['facebook','dblp-A']
	euc_widths = {'facebook':[4], 'dblp-A':[0.5]}
	cos_num_planes = {'facebook':[25], 'dblp-A':[40]}
	thresholds = [0.10]
	LSHs = ['Cosine']
	band_numbers = {'facebook':[4], 'dblp-A':[2]}
	noise_levels = [0.02]
	num_graph = 4
	fname = 'exp_multi_hash_net_final_align'
	for f in folders:
		for noise_level in noise_levels:
			preprocessing(edge_dir = 'Data/'+f+'.edges', save_dir = f, number = num_graph-1, noise_level = noise_level
				, node_dir = 'Data/'+f+'.nodes', node_label = True, weighted_noise = 1.0, findcenter = -1)
			netalign_runner = PureNetAlign(fname)
			#for lsh in LSHs:
			#	for t in thresholds:
			#		netalign_runner.run(f,lsh,t)
			final_runner = PureFinal(fname)
			for lsh in LSHs:
				for t in thresholds:
					final_runner.run(f,lsh,1, all_1 = True)
					#final_runner.run(f,lsh,1)
				
			#preprocessing(edge_dir = 'Data/'+f+'.edges', save_dir = f, number = num_graph-1, noise_level = noise_level
			#	, node_dir = 'Data/'+f+'.nodes')
			hashalign_runner = HashAlign(fname)
			#hashalign_runner.run(folders = [f], band_numbers = band_numbers[f], cos_num_plane = cos_num_planes[f]
			# 		, LSHs = LSHs, thresholds = thresholds, compute_netalign = True)
			hashalign_runner.run(folders = [f], band_numbers = band_numbers[f], cos_num_plane = cos_num_planes[f]
					, LSHs = LSHs, thresholds = thresholds, compute_final = True)
