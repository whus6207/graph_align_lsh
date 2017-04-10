from preprocessing_syn import *
from Experiment_pre import *
import pandas as pd
import pickle

if __name__ == '__main__':
	fname = 'exp_euc.pkl'
	testdata = ['facebook','email']
	df = pd.DataFrame(
		columns=['filename','nodeAttributeFile', 'has_noise', 'GraphType'\
			, 'bandNumber', 'adaptiveLSH', 'LSHType', 'threshold'\
			, 'rank_score', 'rank_score_upper', 'correct_score', 'correct_score_upper', 'correct_score_hungarian'\
			, 'pairs_computed', 'euc_width'])

	eucWidth = [0.2,0.5,1,2,4]
	
	for data in testdata:
		preprocessing(edge_dir='Data/'+ data + '.edges', number=1, noise_level=0.01, weighted_noise=1, save_dir=data)
		for width in eucWidth:
			for i in range(3):
				df = experiment(df, filename=data, bandNumber=2, noise_level=0.01, adaptiveLSH=False, LSHType='Euclidean', euc_width = width, threshold = 0.001)
				df.iloc[-1, df.columns.get_loc('euc_width')] = width
				df.to_csv('exp_euc.csv')

	pickle.dump(df, open(fname,'wb'))
	writer = pd.ExcelWriter('exp_euc.xlsx')
	df.to_excel(writer, sheet_name='Sheet1')
	writer.save()
