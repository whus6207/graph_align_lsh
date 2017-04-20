import matplotlib.pyplot as plt
import pandas as pd 
import snap
from utils.multi_sparse_utils import *



edge_dir = 'Data/facebook.edges'
graph_type = 'Undirected'
number = 1
name = 'fb_dist'

multi_graphs, multi_perm, syn_path = generate_multi_graph_synthetic(filename = edge_dir, graph_type = graph_type, number = number
			, noise_level = 0.01, weighted_noise = 0, weighted = False, is_perm = False)
node_num, n = multi_graphs['M0'].get_shape() 

att = {}
for key, a in multi_graphs.iteritems():
	attributeNames = ['Degree', 'NodeEccentricity']
	attributes = pd.DataFrame(np.zeros((node_num, len(attributeNames))), columns =  attributeNames)
	UGraph = snap.LoadEdgeList(snap.PUNGraph, syn_path + '/' + key +'.edges', 0, 1)
	degree = np.zeros((node_num,))
	OutDegV = snap.TIntPrV()
	snap.GetNodeOutDegV(UGraph, OutDegV)
	for item in OutDegV:
	    degree[item.GetVal1()] = item.GetVal2()
	attributes['Degree'] = degree

	nodeEcc = np.zeros((node_num,))
	for NI in UGraph.Nodes():
	    nodeEcc[NI.GetId()] = snap.GetNodeEcc(UGraph, NI.GetId(), False)
	attributes['NodeEccentricity'] = nodeEcc
	att[key] = attributes
print att

for attr in ['Degree', 'NodeEccentricity']:
	plt.figure()
	bins = np.linspace(min(min(att['M0'][attr]), min(att['M1'][attr])), max(max(att['M0'][attr]), max(att['M1'][attr])), 40)
	plt.hist(att['M0'][attr], bins, alpha = 0.5, label = 'Origin')
	plt.hist(att['M1'][attr], bins, alpha = 0.5, label = 'With noise')
	plt.xlabel(attr, fontsize = 25)
	plt.ylabel('Frequency', fontsize = 25)
	plt.legend(bbox_to_anchor=(0.5, 1.15), prop={'size':20}, loc='upper center', columnspacing = 7.5, ncol=2, shadow = True)
	plt.tick_params(labelsize=20)
	plt.savefig('exp_result/plot/' + name + '_' + attr +'.png', bbox_inches='tight')

