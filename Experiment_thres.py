#import matplotlib.pyplot as plt
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
fname = 'exp_threshold.pkl'
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