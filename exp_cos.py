from preprocessing_syn import *
from Experiment_pre import *
import pandas as pd
import pickle

if __name__ == '__main__':
	fname = 'exp_cos.pkl'
	testdata = ['facebook','email']
	df = pd.DataFrame(
		columns=['filename','nodeAttributeFile', 'has_noise', 'GraphType'\
			, 'bandNumber', 'adaptiveLSH', 'LSHType', 'threshold'\
			, 'rank_score', 'rank_score_upper', 'correct_score', 'correct_score_upper', 'correct_score_hungarian'\
			, 'pairs_computed', 'num_plane'])

	cosBands = [5,10,20,40,70,100]
	
	for data in testdata:
		preprocessing(edge_dir='Data/'+ data + '.edges', number=1, noise_level=0.01, weighted_noise=1, save_dir=data)
		for band in cosBands:
			for i in range(3):
				df = experiment(df, filename=data, bandNumber=2, noise_level=0.01, adaptiveLSH=False, LSHType='Cosine', cos_num_plane = band, threshold = 0.001)
				df.iloc[-1, df.columns.get_loc('num_plane')] = band
				df.to_csv('exp_cos.csv')

	pickle.dump(df, open(fname,'wb'))
	writer = pd.ExcelWriter('exp_cos.xlsx')
	df.to_excel(writer, sheet_name='Sheet1')
	writer.save()
