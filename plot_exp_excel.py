import matplotlib.pyplot as plt
import pandas as pd 
import sys

def plot_exp(exp_path, filename, name, x):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	df = pd.read_excel(exp_path)
	# print df
	# df1 = df[(df['LSHType'] == 'Cosine') & (df['filename'] == filename) & (df['threshold'] == 0.001)]
	df1 = df[(df['LSHType'] == 'Cosine') & (df['filename'] == filename) & (df['bandNumber'] == 6)]
	# df2 = df[(df['LSHType'] == 'Euclidean') & (df['filename'] == filename) & (df['threshold'] == 0.001)]
	print df1
	x_t = df1[x]
	y_s = df1['rank_score']
	y_t = df1['matching_time']

	# x_t2 = df2[x]
	# y_s2 = df2['rank_score']
	# y_t2 = df2['matching_time']
	
	# ax1.plot(x_t, y_s,  '--', marker='x', label = 'cos: rank score')
	ax1.plot(x_t, y_s,  'g--', marker='x', label = 'rank score')
	# ax1.plot(x_t2, y_s2, '--', marker='x', label = 'euc: rank score')
	ax1.set_ylim([0, 1.0])
	
	ax1.legend(loc=5)

	ax2 = ax1.twinx()
	# ax2.plot(x_t, y_t, '-', marker='o', label = 'cos: matching time')
	ax2.plot(x_t, y_t, '-', marker='o', label = 'matching time')
	# ax2.plot(x_t2, y_t2, '-', marker='o', label = 'euc: matching time')

	# ax1.set_title(lsh_type)
	ax1.set_xlabel('threshold')
	ax1.set_ylabel('rank score (1 / position)')
	ax2.set_ylabel('matching time (sec.)')

	ax2.legend(loc=1)
	# plt.show()
	plt.savefig('exp_result/plot/' + name + '.png')

if __name__ == '__main__':
	plot_exp(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])