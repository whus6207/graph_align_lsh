import matplotlib.pyplot as plt
import pandas as pd 
import sys

def plot_exp(exp_path, filename, name, exp_type):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	df = pd.read_csv(exp_path)
	# print df
	# df2 = df[(df['filename'] == filename) & (df['threshold'] == 0.001)]
	df2 = df[(df['filename'] == filename) & (df['threshold'] == 0.0005)]

	if exp_type == 'cosine':
		x = 'num_plane'
		# x1 = df2['num_plane']
		# x2 = df2['num_plane']
	elif exp_type == 'euclidean':
		x = 'euc_width'
		# x1 = df1['euc_width']
		# x2 = df2['euc_width']
	elif exp_type == 'noise_level':
		x = 'noise_level'
	else:
		raise RuntimeError("Invalid type")

	df2_s = df2.groupby([x], as_index=False)['rank_score'].mean()
	print df2_s
	# y_s = df['rank_score']

	# y_t = df['matching_time']

	# print x1
	# print x2
	# print df1['rank_score']
	# ax1.plot(x1, df1['rank_score'], '--', label = '0.001')
	line1 = ax1.plot(df2_s[x], df2_s['rank_score'],  'g--', marker='x', label = 'Rank score')
	
	ax1.legend(handles=line1, loc=1)
	ax1.set_ylim([0, 1.0])
	# ax1.legend(loc='best')
	# ax1.tick_params('y', colors='b')

	ax2 = ax1.twinx()

	df2_t = df2.groupby([x], as_index=False)['matching_time'].mean()
	print df2_t
	# print df1['matching_time']
	# ax2.plot(x1, df1['matching_time'], '-', label = '0.001')
	line2 = ax2.plot(df2_t[x], df2_t['matching_time'], '-', marker='o', label = 'Matching time')

	# ax1.set_title(exp_type)
	ax1.set_xlabel('euclidean width')
	ax1.set_ylabel('rank score (1 / position)')
	ax2.set_ylabel('matching time (sec.)')

	ax2.legend(handles=line2, loc=2)
	# ax2.legend(loc='best')
	# ax2.tick_params('y', colors='r')
	plt.savefig('exp_result/plot/' + name + '.png')
	# fig.tight_layout()

if __name__ == '__main__':
	plot_exp(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
