import matplotlib.pyplot as plt
import numpy as np 
from attr_utils import *
from lsh_utils import *
from io_sparse_utils import *
from multi_sparse_utils import *
from scipy.sparse import identity
from netalign_utils import *
import pandas as pd
#import h5py
import os.path
import pickle
import time

def sim_netalign(df, filename, LSHType = 'Cosine'):
	
	# Load all necessary data
	metadata = {}
	centers = []
	found_center = None
	graph_attrs = {}
	graph_perm = {}
	multi_graphs = {}

	sim_matrix = {}
	matching_matrix = {}
	# Load synthetic graph information
	with open('./private_data/' + filename + '/metadata') as f:
		for line in f:
			line = line.strip().split()
			metadata[line[0]] = line[1]

	# Check multiple graphs
	if metadata['number'] >= 1:	
		with open('./private_data/' + filename + '/centers') as f:
			for line in f:
				centers.append(line.strip().split()[0])
			f.close()
	else:
		raise RuntimeError("Need two graphs to align")

	# Load all graph attributes
	graph_attrs = pickle.load(open('./private_data/' + filename + '/attributes.pkl', 'rb'))
	graph_perm = pickle.load(open('./private_data/' + filename + '/permutations.pkl', 'rb'))
	
	multi_graphs = pickle.load(open('./private_data/' + filename + '/multi_graphs.pkl', 'rb'))

	avg_netalign_score = 0
	matching_time = 0
	netalign_scores = {}
	start_match = time.time()
	for center_id in centers:
		for g in graph_attrs.keys():
			if (center_id, g) not in sim_matrix and (g, center_id) not in sim_matrix and g != center_id:
				if (g, center_id) in sim_matrix:
					sim_matrix[(center_id, g)] = sim_matrix[(g, center_id)]
				else:
					print '!!! computed sim_matrix !!!'
					sim_matrix[(center_id, g)] = computeWholeSimMat(graph_attrs[center_id], graph_attrs[g], LSHType)
				matching_matrix[(center_id, g)] = filter_sim_to_match(sim_matrix[(center_id, g)], 0.2)	

		for g in multi_graphs.keys():
			if g == center_id:
				continue
			if (g, center_id) not in netalign_scores:
				netalign_scores[(center_id, g)] = getNetalignScore(multi_graphs[center_id], multi_graphs[g], matching_matrix[(center_id, g)]
													, graph_perm[center_id], graph_perm[g])
			else:
				netalign_scores[(center_id, g)] = netalign_scores[(g, center_id)]
			avg_netalign_score += netalign_scores[(center_id, g)]

			print "=========================================================="
			print filename + ' ' + g 
			print "GraphType = " + metadata['graph_type'] 
			print "noise_level = " + metadata['noise_level'] + ", nodeAttributeFile = " + metadata['node_dir']
			print "netalign score: %f" %(netalign_scores[(center_id, g)])
	avg_netalign_score /=  (len(multi_graphs.keys())**2 - len(multi_graphs.keys()) )
	matching_time = time.time() - start_match

	df = df.append({'filename':filename, 'nodeAttributeFile': metadata['node_dir']\
		, 'noise_level':metadata['noise_level']\
		, 'avg_netalign_score': avg_netalign_score\
		, 'matching_time': matching_time\
		}, ignore_index=True)

	return df
		

def filter_sim_to_match(sim_matrix, percentage):
	sim_lil = sim_matrix.tolil()
	def max_n_percent(row_data, row_id, n):
		if not n:
			n = 1
		id = row_data.argsort()[-n:]
		top_vals = row_data[id]
		top_ids = row_id[id]
		return top_vals, top_ids, id
	for i in xrange(sim_lil.shape[0]):
		d, r = max_n_percent(np.array(sim_lil.data[i])
				, np.array(sim_lil.rows[i]), int(percentage*sim_lil.shape[1]))[:2]
		sim_lil.data[i]=d.tolist()
		sim_lil.rows[i]=r.tolist()
	sim_matrix = sim_lil.tocsr()
	return sim_matrix

if __name__ == '__main__':

	fname = 'exp_pure_netalign'

	if os.path.isfile(fname+'.pkl'):
		with open(fname+'.pkl', 'rb') as f:
			df = pickle.load(f)
	else:
		df = pd.DataFrame(
			columns=['filename','nodeAttributeFile', 'noise_level', 'GraphType'\
				, 'bandNumber', 'netalign_score'\
				, 'avg_derived_netalign', 'matching_time'])
	
	df = sim_netalign(df,filename='facebook')
	pickle.dump(df, open(fname+'.pkl','wb'))
	df.to_csv(fname+'.csv')
