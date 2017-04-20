import matplotlib.pyplot as plt
import pandas as pd 
import sys

def plot_exp(exp_path, filename, name, lsh):
	plt.figure()

	df = pd.read_csv(exp_path)
	print "hi"
	df['final_score'] = df['final_score'].fillna(0)
	df['netalign_score'] = df['netalign_score'].fillna(0)
	df['isorank_score'] = df['isorank_score'].fillna(0)
	df['avg_netalign_score'] = df['avg_netalign_score'].fillna(0)
	df['avg_isorank_score'] = df['avg_isorank_score'].fillna(0)
	df['avg_final_score'] = df['avg_final_score'].fillna(0)

	# print df
	# df2 = df[(df['filename'] == filename) & (df['threshold'] == 0.001)]
	# df2 = df[(df['filename'] == filename) & (df['threshold'] == 0.0005)]
	if lsh == 'Cosine': 
		df2 = df[(df['filename'] == filename) & (df['LSHType'] == 'Cosine')]
	else:
		df2 = df[(df['filename'] == filename) & (df['LSHType'] == 'Euclidean')]

	df_ha = df2[(df2['netalign_score'] == 0) & (df2['final_score'] == 0) & (df2['isorank_score'] == 0)]
	print df_ha
	df_ha_n = df2[(df2['netalign_score'] != 0) & (df2['avg_netalign_score'] == 0)]
	print df_ha_n
	df_ha_f = df2[(df2['final_score'] != 0) & (df2['avg_final_score'] == 0)]
	print df_ha_f
	df_n = df2[(df2['netalign_score'] != 0)  & (df2['avg_netalign_score'] != 0)]
	print df_n
	df_i = df2[(df2['isorank_score'] != 0)  & (df2['avg_isorank_score'] != 0)]
	print df_i
	df_f = df2[(df2['final_score'] != 0) & (df2['avg_final_score'] != 0)]
	print df_f

	x = 'noise_level'

	plt.plot(df_ha[x], df_ha['correct_score'], '-', color = 'b', linewidth=2.0, marker='*',  markersize=10, label = 'HashAlign')
	plt.plot(df_ha_n[x], df_ha_n['netalign_score'], '-', linewidth=2.0, color = '#040273', marker='o', 
		markersize=10, label = 'HashAlign-NA')
	plt.plot(df_ha_f[x], df_ha_f['final_score'], '-', linewidth=2.0, color = '#069af3', markersize=10, marker='d', label = 'HashAlign-FN')
	plt.plot(df_i[x], df_i['isorank_score'],  '--', linewidth=2.0, color = '#f97306', marker='s', markersize=10, markerfacecolor='None', 
		markeredgecolor='#f97306', markeredgewidth=1.5, label = 'IsoRank')
	plt.plot(df_n[x], df_n['netalign_score'],  '--', linewidth=2.0, color = '#40a368', marker='^', markersize=10, markerfacecolor='None',
	markeredgecolor='#40a368', markeredgewidth=1.5, label = 'NetAlign') # orange
	plt.plot(df_f[x], df_f['final_score'],  '--', color = 'r', marker='x', markeredgewidth=1.5, markersize=10, label = 'FINAL')

	plt.legend(bbox_to_anchor=(0.5, 1.25), prop={'size':20}, loc='upper center', columnspacing = 0.5, ncol=3, shadow = True)
	plt.ylim([-0.05, 1.1])

	# ax1.legend(loc='best')
	# ax1.tick_params('y', colors='b')

	# print df1['matching_time']
	# ax2.plot(x1, df1['matching_time'], '-', label = '0.001')
	# line2 = ax2.plot(df2_t[x], df2_t['matching_time'], '-', marker='o', label = 'Matching time')

	# ax1.set_title(exp_type)
	plt.tick_params(labelsize=20)
	plt.xlabel('Noise level ', fontsize=25)
	plt.ylabel('Accuracy', fontsize = 25)
	# ax2.set_ylabel('matching time (sec.)')

	# ax2.legend(handles=line2, loc=4)
	# ax2.legend(loc='best')
	# ax2.tick_params('y', colors='r')
	plt.savefig('exp_result/plot/' + name + '.png', bbox_inches='tight')
	# fig.tight_layout()

if __name__ == '__main__':
	plot_exp(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
