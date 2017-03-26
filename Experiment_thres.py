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
bandNumber = [2,4,8]
noise_level = [0.02]
gfiles = [['facebook/0.edges', 'Undirected'], ['metadata/phys.edges', 'Directed'],
		 ['metadata/email.edges', 'Undirected']]
fname = 'exp_threshold_netal.pkl'
thresholds = [1,2,4,8]

if os.path.isfile(fname):
	with open(fname, 'rb') as f:
		df = pickle.load(f)
else:
	df = pd.DataFrame(
		columns=['filename','nodeAttributeFile', 'is_perm', 'has_noise', 'noise_level'\
			, 'GraphType', 'bandNumber', 'adaptiveLSH', 'LSHType', 'threshold'\
			, 'rank_score', 'rank_score_upper', 'correct_score', 'correct_score_upper', 'correct_score_hungarian'\
			, 'pairs_computed'])

a = adaptiveLSH[0]
n = noise[0]
l = noise_level[0]
for gfile in gfiles:
	for b in bandNumber:
		for thr in thresholds:
			if thr > b:
				continue
			df = experiment(df, filename = gfile[0], nodeAttributeFile = None, 
				multipleGraph = False, is_perm = False, has_noise = n, noise_level = l,
				plotAttribute = False, 
				plotBucket = False, plotCorrectness = False, GraphType = gfile[1], bandNumber = b, 
				adaptiveLSH = a, LSHType = 'Cosine', compute_sim = True, compute_hungarian = False,
				loop_num = 3, threshold = thr)
			df = experiment(df, filename = gfile[0], nodeAttributeFile = None, 
				multipleGraph = False, is_perm = False, has_noise = n, noise_level = l,
				plotAttribute = False, 
				plotBucket = False, plotCorrectness = False, GraphType = gfile[1], bandNumber = b, 
				adaptiveLSH = a, LSHType = 'Euclidean', compute_sim = True, compute_hungarian = False,
				loop_num = 3, threshold = thr)
			pickle.dump(df, open(fname,'wb'))

writer = pd.ExcelWriter('exp_threshold.xlsx')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()

### for generating color map
import numpy as np
import matplotlib.colors as mcolors

cm = plt.cm.get_cmap('RdYlBu')
#for fname in ['facebook/0.edges','metadata/phys.edges','metadata/email.edges']:
for fname in ['facebook/0.edges','metadata/email.edges']:
	ff = fname.strip().split('/')[1]
	#fig = plt.figure()
	#ax = fig.add_subplot(111)
	for bn in [2,4,8]:
		for lsh, mark in zip(['Cosine', 'Euclidean'], ['^','o']):
			df_sub = df[(df['LSHType']==lsh) & (df['filename']==fname) & (df['bandNumber']==bn)]\
				.loc[:,['threshold', 'rank_score','rank_score_upper'\
				, 'correct_score_hungarian', 'pairs_computed', 'netalign_correct_score']]
			df_sub = df_sub.dropna(thresh=1)
			x = df_sub['pairs_computed'].values
			net_score = df_sub['netalign_correct_score'].values
			rank = df_sub['rank_score'].values
			for i in range(len(x)):
				plt.annotate(str(bn), (x[i], net_score[i]))
			plt.scatter(x,net_score, marker=mark, s=50, c=df_sub['threshold'].values/float(bn), vmin=0, vmax=1, cmap=cm)
	plt.xlabel('pairs computed')
	plt.ylabel('netalign score ')
	plt.colorbar()
	plt.suptitle('netalign score to pairs computed with threshold('+fname+')')
	plt.show()
	#fig.savefig(ff+'_threshold.png'\
	#	, bbox_extra_artists=(lgd,), bbox_inches='tight')

for fname in ['facebook/0.edges','metadata/phys.edges','metadata/email.edges']:
	ff = fname.strip().split('/')[1]
	lines = []
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for bn in [2,4,8]:
		for lsh in ['Cosine', 'Euclidean']:
			df_sub = df[(df['LSHType']==lsh) & (df['filename']==fname) & (df['bandNumber']==bn)]\
				.loc[:,['threshold', 'rank_score','rank_score_upper'\
				, 'correct_score_hungarian', 'pairs_computed']]
			x = df_sub['threshold'].values
			rank = df_sub['rank_score'].values
			#line, = plt.plot(x,rank, label='rank_score bandNum='+str(bn)+'('+lsh+')', linewidth=2)
			ax.plot(x,rank, label='rank_score bandNum='+str(bn)+'('+lsh+')', linewidth=2)
			lines.append(line)
	handles ,labels = ax.get_legend_handles_labels()
	lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
	ax.grid('on')
	plt.xlabel('threshold')
	#plt.legend(handles=lines, loc='best')
	plt.suptitle('Acc to threshold for different bandNumber('+fname+')')
	#plt.show()
	fig.savefig(ff+'_threshold.png'\
		, bbox_extra_artists=(lgd,), bbox_inches='tight')

for fname in ['facebook/0.edges','metadata/phys.edges','metadata/email.edges']:
	ff = fname.strip().split('/')[1]
	lines = []
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for bn in [2,4,8]:
		for lsh in ['Cosine', 'Euclidean']:
			df_sub = df[(df['LSHType']==lsh) & (df['filename']==fname) & (df['bandNumber']==bn)]\
				.loc[:,['threshold', 'rank_score','rank_score_upper'\
				, 'correct_score_hungarian', 'pairs_computed']]
			x = df_sub['threshold'].values
			pairs = df_sub['pairs_computed'].values
			#line, = plt.plot(x,rank, label='rank_score bandNum='+str(bn)+'('+lsh+')', linewidth=2)
			ax.plot(x,pairs, label='pairs computed bandNum='+str(bn)+'('+lsh+')', linewidth=2)
			#lines.append(line)
	handles ,labels = ax.get_legend_handles_labels()
	lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
	ax.grid('on')
	plt.xlabel('threshold')
	#plt.legend(handles=lines, loc='best')
	plt.suptitle('Pairs computed for different bandNumber('+fname+')')
	#plt.show()
	fig.savefig(ff+'_threshold_pair.png'\
		, bbox_extra_artists=(lgd,), bbox_inches='tight')

