from preprocessing_syn import *
from HashAlign import *
import pandas as pd
import pickle

if __name__ == '__main__':
	fname = 'exp_cos_dblp_attr.pkl'
	testdata = ['dblp-A']
	if os.path.isfile(fname):
		with open(fname, 'rb') as f:
			df = pickle.load(f)
	else:
		df = pd.DataFrame(
			columns=['filename','nodeAttributeFile', 'has_noise', 'GraphType'\
			, 'bandNumber', 'adaptiveLSH', 'LSHType', 'threshold'\
			, 'rank_score', 'rank_score_upper', 'correct_score', 'correct_score_upper', 'correct_score_hungarian'\
			, 'pairs_computed', 'num_plane'])

	cosBands = [100]
	bandnum = [1,2]	
	for data in testdata:
		preprocessing(edge_dir='Data/dblp/'+ data + '.edges', node_dir = 'Data/dblp/'+ data + '.attr',number=3, noise_level=0.02, save_dir=data)
		for band in cosBands:
			for thres in [0.001]:
				for bn in bandnum:
					df = experiment(df, filename=data, bandNumber=bn, compute_sim=False, compute_netalign = True
						, noise_level=0., adaptiveLSH=False, LSHType='Cosine'
						, cos_num_plane = band, threshold = thres)
					df.iloc[-1, df.columns.get_loc('num_plane')] = band
					df.to_csv('exp_cos_dblp_attr.csv')
					pickle.dump(df, open(fname,'wb'))
