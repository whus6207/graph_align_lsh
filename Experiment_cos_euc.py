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
fname = 'exp_result_cos_euc_email.pkl'

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
				df = experiment(df, filename = 'metadata/email.edges', nodeAttributeFile = None, 
					multipleGraph = False, is_perm = False, has_noise = n, noise_level = l,
					plotAttribute = False, 
					plotBucket = False, plotCorrectness = False, GraphType = 'Undirected', bandNumber = b, 
					adaptiveLSH = a, LSHType = 'Cosine', compute_sim = True, compute_hungarian = False,
					loop_num = 1)
				df = experiment(df, filename = 'metadata/email.edges', nodeAttributeFile = None, 
					multipleGraph = False, is_perm = False, has_noise = n, noise_level = l,
					plotAttribute = False, 
					plotBucket = False, plotCorrectness = False, GraphType = 'Undirected', bandNumber = b, 
					adaptiveLSH = a, LSHType = 'Euclidean', compute_sim = True, compute_hungarian = False,
					loop_num = 1)
				if a:
					break

pickle.dump(df, open(fname,'wb'))

writer = pd.ExcelWriter('exp_result_cos_euc_email.xlsx')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()

### draw for attributed graph

# df_cos = df[(df['LSHType']=='Cosine') & (df['nodeAttributeFile']=='phys.nodes')].loc[:,['bandNumber',\
# 	'rank_score', 'correct_score_hungarian', 'pairs_computed']]
# x_cos = df_cos['bandNumber'].values
# rank_cos = df_cos['rank_score'].values
# hung_cos = df_cos['correct_score_hungarian'].values
# pair_cos = df_cos['pairs_computed'].values

# line1, = plt.plot(x_cos, rank_cos, label="rank_score(cos)", linestyle='--')
# line2, = plt.plot(x_cos, hung_cos, label="correct score hungarian(cos)", linewidth=2)
# line3, = plt.plot(x_cos, pair_cos, label="% pairs computed(cos)", linewidth=2)

# df_euc = df[(df['LSHType']=='Euclidean') & (df['nodeAttributeFile']=='phys.nodes')].loc[:,['bandNumber',\
# 	'rank_score', 'correct_score_hungarian', 'pairs_computed']]
# x_euc = df_euc['bandNumber'].values
# rank_euc = df_euc['rank_score'].values
# hung_euc = df_euc['correct_score_hungarian'].values
# pair_euc = df_euc['pairs_computed'].values
# line4, = plt.plot(x_cos, rank_euc, label="rank_score(euc)", linestyle='--')
# line5, = plt.plot(x_cos, hung_euc, label="correct score hungarian(euc)", linewidth=2)
# line6, = plt.plot(x_cos, pair_euc, label="% pairs computed(euc)", linewidth=2)

# plt.xlabel('bandNumber')
# plt.legend(handles=[line1, line2, line3, line4, line5, line6], loc='best')
# plt.suptitle('LSH Comparison(Cosine vs Euclidean)')
# plt.show()

### draw for unattributed graph

df_cos = df[(df['LSHType']=='Cosine') & (df['nodeAttributeFile']=='None')].loc[:,['bandNumber',\
	'rank_score','rank_score_upper', 'correct_score_hungarian', 'pairs_computed']]
#df_cos =df_cos.groupby(['bandNumber']).mean()
x_cos = df_cos['bandNumber'].values
#x_cos = df_cos.index.values
rank_cos = df_cos['rank_score'].values
rank_upper_cos = df_cos['rank_score_upper'].values*2
hung_cos = df_cos['correct_score_hungarian'].values
pair_cos = df_cos['pairs_computed'].values

line1, = plt.plot(x_cos, rank_cos, label="rank_score(cos)", linestyle='--')
line2, = plt.plot(x_cos, rank_upper_cos, label='rank_score_upper(cos)', linewidth=2)
#line3, = plt.plot(x_cos, hung_cos, label="correct score hungarian(cos)", linewidth=2)
line4, = plt.plot(x_cos, pair_cos, label="% pairs computed(cos)", linewidth=2)

df_euc = df[(df['LSHType']=='Euclidean') & (df['nodeAttributeFile']=='None')].loc[:,['bandNumber',\
	'rank_score','rank_score_upper', 'correct_score_hungarian', 'pairs_computed']]
#df_euc =df_euc.groupby(['bandNumber']).mean()
x_euc = df_euc['bandNumber'].values
#x_euc = df_euc.index.values
rank_euc = df_euc['rank_score'].values
rank_upper_euc = df_euc['rank_score_upper'].values*2
hung_euc = df_euc['correct_score_hungarian'].values
pair_euc = df_euc['pairs_computed'].values
line5, = plt.plot(x_cos, rank_euc, label="rank_score(euc)", linestyle='--')
line6, = plt.plot(x_cos, rank_upper_euc, label='rank_score_upper(euc)', linewidth=2)
#line7, = plt.plot(x_cos, hung_euc, label="correct score hungarian(euc)", linewidth=2)
line8, = plt.plot(x_cos, pair_euc, label="% pairs computed(euc)", linewidth=2)

plt.xlabel('bandNumber')
#plt.legend(handles=[line1, line2, line3, line4, line5, line6], loc='best')
plt.legend(handles=[line1, line2, line4, line5, line6, line8], loc='best')
plt.suptitle('LSH Comparison(Cosine vs Euclidean)')
plt.show()

