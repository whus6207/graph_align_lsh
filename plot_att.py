from multi_utils import generate_multi_graph_synthetic
from attr_utils import *
import matplotlib.pyplot as plt


path = 'metadata/multigraph/Undirected'
multi_graphs = generate_multi_graph_synthetic(filename = 'facebook/0.edges', graph_type = 'Undirected')
graph_attrs = {}
for key in multi_graphs.keys():
	attributesA = getUndirAttribute(path + '/' + key)
	graph_attrs[key] = attributesA

n = len(graph_attrs)
attributes_all = attributesA.columns[2:]
for attr in attributes_all:


	print attr
	# first_att = graph_attrs.itervalues().next()
	min_bin = min(graph_attrs['M0.edges'][attr])
	max_bin = max(graph_attrs['M0.edges'][attr])
	# for key, attributes in graph_attrs.iteritems():
	# 	min_bin = min(min(attributes[attr]), min_bin)
	# 	max_bin = max(max(attributes[attr]), max_bin)
	# f, ax = plt.subplots(n, 1, sharex=True, sharey=True)
	for key in graph_attrs.keys():
		plt.clf()
		plt.figure()
		if key == 'M0.edges':
			continue
		# plt.subplot(n, 1, i+1, sharex=True, sharey=True)
		bins = np.linspace(min(min(graph_attrs[key][attr]), min_bin), max(max(graph_attrs[key][attr]), max_bin), 40)
		plt.hist(graph_attrs['M0.edges'][attr], bins, alpha = 0.5, label = 'M0.edges')
		plt.hist(graph_attrs[key][attr], bins, alpha = 0.5, label = key)
		# ax[i].hist(attributes[attr], bins, label = key)

		plt.xlabel(attr)
		plt.ylabel('Frequency')
		plt.legend(loc = 'best')
		plt.title('Distribution of ' + attr)
		plt.savefig('./att_fig/multi_noise/' + attr + '_' + key[:2] + '.png')
