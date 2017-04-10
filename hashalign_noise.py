from preprocessing_syn import *
from Experiment_pre import *
import pandas as pd
import pickle

if __name__ == '__main__':
	fname = 'exp_noise.pkl'
	testdata = ['facebook','email']
	df = pd.DataFrame(
		columns=['filename','nodeAttributeFile', 'has_noise', 'GraphType'\
			, 'bandNumber', 'adaptiveLSH', 'LSHType', 'threshold'\
			, 'rank_score', 'rank_score_upper', 'correct_score', 'correct_score_upper', 'correct_score_hungarian'\
			, 'pairs_computed'])
	noises = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
	for data in testdata:
		for noise in noises:
			for i in range(3):
				preprocessing(edge_dir='Data/'+data+'.edges', number=1, noise_level=noise, weighted_noise=1, save_dir=data)	
				df = experiment(df, filename=data, bandNumber=2, adaptiveLSH=False, LSHType='Cosine')
				df.to_csv('exp_noise.csv')

	pickle.dump(df, open(fname,'wb'))
	writer = pd.ExcelWriter('exp_noise.xlsx')
	df.to_excel(writer, sheet_name='Sheet1')
	writer.save()
