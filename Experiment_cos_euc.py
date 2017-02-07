import matplotlib.pyplot as plt
import numpy as np 
from attr_utils import *
from io_utils import *
from lsh_utils import *
from Experiment import experiment
import pandas as pd
import os.path
import pickle

adaptiveLSH = [False]
noise = [True]
bandNumber = [2,3,4,8]
noise_level = [0.02]
fname = 'exp_result_cos_euc.pkl'

if os.path.isfile(fname):
	with open(fname, 'rb') as f:
		df = pickle.load(f)
else:
	df = pd.DataFrame(
		columns=['filename','nodeAttributeFile', 'is_perm', 'has_noise', 'GraphType', 'bandNumber', 'adaptiveLSH', 'LSHType'\
			, 'rank_score', 'rank_score_upper', 'correct_score', 'correct_score_upper', 'correct_score_hungarian'\
			, 'pairs_computed'])


for a in adaptiveLSH:
	for n in noise:
		for b in bandNumber:
			for l in noise_level:
			# df = experiment(df, filename = 'metadata/A.edges', nodeAttributeFile = None, 
			# 	multipleGraph = False, is_perm = False, has_noise = n, plotAttribute = False, 
			# 	plotBucket = False, plotCorrectness = False, GraphType = 'Undirected', bandNumber = b, 
			# 	adaptiveLSH = a, LSHType = 'Cosine')
			# df = experiment(df, filename = 'metadata/A.edges', nodeAttributeFile = None, 
			# 	multipleGraph = False, is_perm = False, has_noise = n, plotAttribute = False, 
			# 	plotBucket = False, plotCorrectness = False, GraphType = 'Undirected', bandNumber = b, 
			# 	adaptiveLSH = a, LSHType = 'Euclidean')
				df = experiment(df, filename = 'metadata/phys.edges', nodeAttributeFile = None, 
					multipleGraph = False, is_perm = False, has_noise = n, noise_level = l,
					plotAttribute = False, 
					plotBucket = False, plotCorrectness = False, GraphType = 'Directed', bandNumber = b, 
					adaptiveLSH = a, LSHType = 'Cosine')
				df = experiment(df, filename = 'metadata/phys.edges', nodeAttributeFile = None, 
					multipleGraph = False, is_perm = False, has_noise = n, noise_level = l,
					plotAttribute = False, 
					plotBucket = False, plotCorrectness = False, GraphType = 'Directed', bandNumber = b, 
					adaptiveLSH = a, LSHType = 'Euclidean')
				df = experiment(df, filename = 'metadata/phys.edges', nodeAttributeFile = 'phys.nodes', 
					multipleGraph = False, is_perm = False, has_noise = n, noise_level = l,
					plotAttribute = False, 
					plotBucket = False, plotCorrectness = False, GraphType = 'Directed', bandNumber = b, 
					adaptiveLSH = a, LSHType = 'Cosine')
				df = experiment(df, filename = 'metadata/phys.edges', nodeAttributeFile = 'phys.nodes', 
					multipleGraph = False, is_perm = False, has_noise = n, noise_level = l,
					plotAttribute = False, 
					plotBucket = False, plotCorrectness = False, GraphType = 'Directed', bandNumber = b, 
					adaptiveLSH = a, LSHType = 'Euclidean')
				if a:
					break

pickle.dump(df, open(fname,'wb'))

writer = pd.ExcelWriter('exp_result_cos_euc.xlsx')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()

df_cos = df[(df['LSHType']=='Cosine') & (df['nodeAttributeFile']=='phys.nodes')].loc[:,['bandNumber',\
	'rank_score', 'correct_score_hungarian', 'pairs_computed']]
x_cos = df_cos['bandNumber'].values
rank_cos = df_cos['rank_score'].values
hung_cos = df_cos['correct_score_hungarian'].values
pair_cos = df_cos['pairs_computed'].values

line1, = plt.plot(x_cos, rank_cos, label="rank_score(cos)", linestyle='--')
line2, = plt.plot(x_cos, hung_cos, label="correct score hungarian(cos)", linewidth=2)
line3, = plt.plot(x_cos, pair_cos, label="% pairs computed(cos)", linewidth=2)

df_euc = df[(df['LSHType']=='Euclidean') & (df['nodeAttributeFile']=='phys.nodes')].loc[:,['bandNumber',\
	'rank_score', 'correct_score_hungarian', 'pairs_computed']]
x_euc = df_euc['bandNumber'].values
rank_euc = df_euc['rank_score'].values
hung_euc = df_euc['correct_score_hungarian'].values
pair_euc = df_euc['pairs_computed'].values
line4, = plt.plot(x_cos, rank_euc, label="rank_score(euc)", linestyle='--')
line5, = plt.plot(x_cos, hung_euc, label="correct score hungarian(euc)", linewidth=2)
line6, = plt.plot(x_cos, pair_euc, label="% pairs computed(euc)", linewidth=2)

plt.legend(handles=[line1, line2, line3, line4, lin5, lin6])
plt.suptitle('LSH Comparison(Cosine vs Euclidean)')
plt.show()

