import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder

import pandas as pd 
from seaborn import load_dataset

from boruta_py import BorutaPy, BatchBorutaPy
from tracking_boruta_py import TrackingBorutaPy, TrackingBatchBorutaPy

if __name__ == '__main__':

	rand_seed = np.random.randint(1e9)
	np.set_printoptions(linewidth=360)

	dataset = load_dataset('iris')
	X = dataset.iloc[:, :-1]
	y = dataset.iloc[:, -1]
	X = np.hstack((X.values, np.random.random(X.shape)))
	X = np.hstack((X, np.random.random(X.shape)))

	# dataset = load_dataset('diamonds')
	# y = dataset.color.values
	# dataset = dataset.drop('color', axis=1)
	# X = pd.get_dummies(dataset, columns=['cut', 'clarity']) 
	# X = np.hstack((X, np.random.random(X.shape)))

	start = time.time()
	b = TrackingBorutaPy(
		RandomForestClassifier(verbose=0, n_jobs=-1),
		n_estimators='auto',
		two_step=False,
		random_state=rand_seed,
		max_iter=100,
		verbose=1)
	fit_res_b = b.fit(X, y) 
	time_b = time.time() - start

	# b100 = BorutaPy(
	# 	RandomForestClassifier(verbose=0),
	# 	n_estimators=100,
	# 	two_step=False,
	# 	verbose=1)
	# fit_res_b100 = b100.fit(X, y) 

	# bb = BatchBorutaPy(
	# 	[
	# 		RandomForestClassifier(verbose=0), 
	# 		RandomForestClassifier(verbose=0)
	# 	], 
	# 	n_estimators='auto', #[10, 30]
	# 	two_step=False,
	# 	verbose=1)
	# fit_res_bb = bb.fit(X, y)

	start = time.time()
	bb_i = TrackingBatchBorutaPy(
		[
			RandomForestClassifier(verbose=0, n_jobs=-1),
			ExtraTreesClassifier(verbose=0, n_jobs=-1)
		], 
		n_estimators='auto',
		two_step=False,
		mode='iterative',
		verbose=1,
		random_state=rand_seed)
	fit_res_bb_i = bb_i.fit(X, y)
	time_bb_i = time.time() - start

	# start = time.time()
	# bb_mp = BatchBorutaPy(
	# 	[
	# 		RandomForestClassifier(verbose=0, n_jobs=-1), 
	# 		# GradientBoostingClassifier(verbose=1),
	# 		ExtraTreesClassifier(verbose=0, n_jobs=-1)
	# 	], 
	# 	n_estimators='auto',
	# 	two_step=False,
	# 	mode='mp',
	# 	n_jobs=2,
	# 	verbose=1,
	# 	random_state=rand_seed)
	# fit_res_bb_mp = bb_mp.fit(X, y)
	# time_bb_mp = time.time() - start

	print(b.support_)
	print(time_b)
	# print(fit_res_b100.support_)
	# print(fit_res_bb.support_)
	print(bb_i.support_)
	print(time_bb_i)
	# print(fit_res_bb_mp.support_)
	# print(time_bb_mp)