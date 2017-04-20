import matplotlib.pyplot as plt
import pandas as pd 
import sys

def plot_exp(exp_path, filename, name, x):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	df = pd.read_excel(exp_path)
	# print df
	df1 = df[(df['LSHType'] == 'Cosine') & (df['filename'] == filename) & (df['threshold'] == 0.001)]
	# df1 = df[(df['LSHType'] == 'Cosine') & (df['filename'] == filename) & (df['bandNumber'] == 6)]
	df2 = df[(df['LSHType'] == 'Euclidean') & (df['filename'] == filename) & (df['threshold'] == 0.001)]
	print df1
	x_t = df1[x]
	y_s = df1['rank_score']
	y_t = df1['matching_time']

	x_t2 = df2[x]
	y_s2 = df2['rank_score']
	y_t2 = df2['matching_time']
	
	ax1.plot(x_t, y_s,  '-', color = 'b', linewidth=2.0,  
		markeredgewidth=1.5, markersize=10, marker='x', label = 'Cos: Rank score')
	# ax1.plot(x_t, y_s,  'g--', marker='x', label = 'rank score')
	ax1.plot(x_t2, y_s2, '-', color = '#069af3', linewidth=2.0,  
		markeredgewidth=1.5, markersize=10, marker='x', label = 'Euc: Rank score')
	ax1.set_ylim([0, 1.0])
	
	ax1.legend(bbox_to_anchor=(0.25, 1.2), prop={'size':16}, loc='upper center', columnspacing = 0.5, shadow = True)
	ax1.tick_params(labelsize=18)
	ax2 = ax1.twinx()
	ax2.plot(x_t, y_t, '--', color = '#033500', linewidth=2.0, 
		markeredgecolor = '#033500', markeredgewidth=1.5, markersize=10, markerfacecolor='None', marker='o', label = 'Cos: Matching time')
	# ax2.plot(x_t, y_t, '-', marker='o', label = 'matching time')
	ax2.plot(x_t2, y_t2, '--', color = '#b0dd16', linewidth=2.0, 
		markeredgecolor = '#b0dd16', markeredgewidth=1.5, markersize=10, markerfacecolor='None', marker='o', label = 'Euc: Matching time')

	# ax1.set_title(lsh_type)
	ax1.set_xlabel('Band number', fontsize = 25)
	ax1.set_ylabel('Rank score (1 / position)', fontsize = 25)
	ax2.set_ylabel('Matching time (sec.)', fontsize = 25)

	ax2.legend(bbox_to_anchor=(0.72, 1.2), prop={'size':16}, loc='upper center', columnspacing = 0.5, shadow = True)
	ax2.tick_params(labelsize=18)
	# plt.show()
	plt.savefig('exp_result/plot/' + name + '.png', bbox_inches='tight')

if __name__ == '__main__':
	plot_exp(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])