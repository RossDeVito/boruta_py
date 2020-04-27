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

	max_df = selector_df.groupby(['model_name', 'finish_time']).max().reset_index()

	last_df = selector_df.groupby(['test', 'model_name', 'finish_time']).last().reset_index()

	to_best = []

	for _, row in last_df.iterrows():
		test_df = selector_df.loc[
			(selector_df.model_name == row.model_name)
			& (selector_df.finish_time == row.finish_time)
		]

		end_useful = test_df.jaccard_useful.values[-1]
		end_useless = test_df.jaccard_useless.values[-1]

		to_best.append(
			test_df.loc[
				(test_df.jaccard_useful == end_useful)
				& (test_df.jaccard_useless == end_useless)
			].min()
		)

	to_best_df = pd.DataFrame(to_best)
	to_best_df = to_best_df[
		(to_best_df.test == 'hc2')
		| (to_best_df.test == 'madelon2')
		| (to_best_df.test == 'madelon4')
	]

	to_best_df = to_best_df[
		(to_best_df.batch)
		| (to_best_df['model_name'].isin(
			to_best_df[to_best_df.test == 'hc2'].model_name.unique())
		)
	]

	melt_df = to_best_df.melt(
		id_vars = [
			'model_name',
			'finish_time',
			'test',
			'env',
			'batch',
			'underlying_model',
			'model_class',
			'time'
		],
		value_vars = [
			'jaccard_useful',
			'jaccard_useless',
		],
		var_name='metric',
		value_name='value'
	)

	sns.set_style('darkgrid')


	sns.relplot(x='time', y='value', hue='model_class', style='batch', 
				row='test', col='metric', 
				data=melt_df,#[melt_df.env == 'laptop'], 
				height=3,
				facet_kws={
					'margin_titles': True,
					'sharey': "row"
				}, 
				alpha=1, edgecolor=None,
				linewidth=2.0, markers=["$\u20DD$", "$\u00D7$"]
			)

	plt.suptitle("Selection on 8-Core Laptop", x=.98, y=.99, ha='right')
	plt.show()


	sns.relplot(x='time', y='value', hue='model_class', style='batch', 
				row='test', col='metric', 
				data=melt_df[melt_df.env == 'ganxis'], 
				height=3,
				facet_kws={
					'margin_titles': True,
					'sharey': "none"
				}, 
				alpha=1, edgecolor=None,
				linewidth=2.0, markers=["$\u20DD$", "$\u00D7$"]
			)

	plt.suptitle("Selection on 128-Core Server", x=.98, y=.99, ha='right')
	plt.show()

	# sns.relplot(x='time', y='value', hue='model_name', style='env', 
	# 			row='test', col='metric', 
	# 			data=melt_df,
	# 			height=3,
	# 			facet_kws={
	# 				'margin_titles': True,
	# 				'sharey': "row"
	# 			}, 
	# 			alpha=1, edgecolor=None,
	# 			linewidth=2.0, markers=["$\u20DD$", "$\u00D7$"]
	# 		)

	# plt.suptitle("Selection on Both", x=.98, y=.99, ha='right')
	# plt.show()
	# plt.close()

	best = to_best_df.groupby('model_name').mean()  
	best_u = best.sort_values('jaccard_useful', ascending=False)
	best_l = best.sort_values('jaccard_useless', ascending=False) 
	

	max_df = max_df[
		(max_df.test == 'hc2')
		| (max_df.test == 'madelon2')
		| (max_df.test == 'madelon4')
	]

	max_df = max_df[
		(max_df.batch)
		| (max_df['model_name'].isin(
			max_df[max_df.test == 'hc2'].model_name.unique())
		)
	]

	melt_df = max_df.melt(
		id_vars = [
			'model_name',
			'finish_time',
			'test',
			'env',
			'batch',
			'underlying_model',
			'model_class',
			'time'
		],
		value_vars = [
			'jaccard',
			'precision',
			'recall',
			'f1',
			'completeness'
		],
		var_name='metric',
		value_name='value'
	)
	
	sns.relplot(x='time', y='value', hue='model_class', style='batch', 
				row='test', col='metric', 
				data=melt_df,
				height=3,
				facet_kws={
					'margin_titles': True,
					'sharey': "row"
				}, 
				alpha=1, edgecolor=None,
				linewidth=2.0, markers=["$\u20DD$", "$\u00D7$"]
			)

	plt.suptitle("Performance Metrics", x=.98, y=.99, ha='right')
	plt.show()

	# g = sns.FacetGrid(melt_df, col='metric', row='test', hue='model_class')
	# g = (g.map(plt.scatter, "time", "value",  alpha=.6)
    #   		.add_legend())

	# plt.show()