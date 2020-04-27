import glob
import os

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

if __name__ == "__main__":

	load_dir = 'run_res'

	filenames = glob.glob(os.path.join(load_dir, '*.csv'))

	dfs = []
	for filename in filenames:
		dfs.append(pd.read_csv(filename))

	selector_df = pd.concat(dfs, ignore_index=True)

	# selector_df_end = selector_df.copy()

	# for test in selector_df.test.unique():

	# 	test_df = selector_df[selector_df['test'] == test]
	# 	test_max_time = test_df.time.max()

	# 	for _, row in test_df.groupby(['model_name', 'finish_time']).max().reset_index().iterrows():
	# 		name = row.model_name
	# 		f_time = row.finish_time

	# 		selector_df_end = selector_df_end.append(
	# 			{
	# 				'model_name': name,
	# 				'finish_time': f_time,
	# 				'time': test_max_time, 
	# 				'model_class': row.model_class,
	# 				'jaccard_useful': row.jaccard_useful,
	# 				'jaccard_useless': row.jaccard_useless,
	# 			}, 
	# 			ignore_index=True
	# 		)  
			

	sns.lineplot(x='time', y='jaccard_useless', hue='model_class', 
		 data=selector_df_end)#, ci=None, estimator=None) style='model_class',

	# sns.pointplot(x='time', y='jaccard_useless', hue='model_class', 
	# 	data=selector_df_end)    

	# plt.show()