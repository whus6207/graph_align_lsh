from preprocessing_syn import *
from HashAlign import *
import pandas as pd
import pickle

if __name__ == '__main__':
	fname = 'exp_cos_dblp_attr.pkl'
	testdata = ['dblp']
	df = pd.DataFrame(
		columns=['filename','nodeAttributeFile', 'noise_level', 'GraphType'\
			, 'bandNumber', 'adaptiveLSH', 'LSHType', 'threshold'\
			, 'rank_score', 'rank_score_upper', 'correct_score', 'correct_score_upper', 'netalign_score'\
			, 'final_score', 'pairs_computed', 'matching_time'])
	
	for data in testdata:
		#preprocessing(edge_dir='Data/'+ data + '.edges', number=1, noise_level=0.01, weighted_noise=1, save_dir=data)
		for band in cosBands:
			for thres in [0.0005, 0.001]:
				df = experiment(df, filename=data, bandNumber=2, compute_sim=False,  noise_level=0.01, adaptiveLSH=False, LSHType='Cosine', cos_num_plane = band, threshold = thres)
			df.iloc[-1, df.columns.get_loc('num_plane')] = band
			df.to_csv('exp_cos_dblp_attr.csv')

	pickle.dump(df, open(fname,'wb'))
	writer = pd.ExcelWriter('exp_cos_dblp.xlsx')
	df.to_excel(writer, sheet_name='Sheet1')
	writer.save()
