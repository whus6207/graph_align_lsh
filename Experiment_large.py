import matplotlib.pyplot as plt
import numpy as np 
from attr_utils import *
from io_utils import *
from lsh_utils import *
from Experiment import experiment
import pandas as pd
import os.path
import pickle

def a():
	print "aaa"

adaptiveLSH = [False]
noise = [True]
bandNumber = [2,4,8]
noise_level = [0.002]
fname = 'exp_result_email.pkl'

if os.path.isfile(fname):
	with open(fname, 'rb') as f:
		df = pickle.load(f)
else:
	df = pd.DataFrame(
		columns=['filename', 'is_perm', 'has_noise', 'noise_level', 'GraphType', 'bandNumber', 'adaptiveLSH', 'LSHType'\
			, 'rank_score', 'rank_score_upper', 'correct_score', 'correct_score_upper', 'correct_score_hungarian'\
			, 'pairs_computed'])

for a in adaptiveLSH:
	for n in noise:
		for b in bandNumber:
			for l in noise_level:
				df = experiment(df, filename = './email/email.edges', multipleGraph = False, largeGraph = True, is_perm = False, 
					has_noise = n, noise_level = l, plotAttribute = False, plotBucket = False, plotCorrectness = False, 
					GraphType = 'Undirected', bandNumber = b, adaptiveLSH = a, LSHType = 'Cosine')
				df = experiment(df, filename = './email/email.edges', multipleGraph = False, largeGraph = True, is_perm = False, 
					has_noise = n, noise_level = l, plotAttribute = False, plotBucket = False, plotCorrectness = False, 
					GraphType = 'Undirected', bandNumber = b, adaptiveLSH = a, LSHType = 'Euclidean')

pickle.dump(df, open(fname,'wb'))

# write to excel
writer = pd.ExcelWriter('exp_result_email.xlsx')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()