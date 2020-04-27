import os
import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import jaccard_score, precision_score, recall_score
from sklearn.metrics import f1_score, completeness_score

from tracking_boruta_py import TrackingBorutaPy, TrackingBatchBorutaPy
from bb_util import get_test_data


def get_bool_jaccard(arr1, arr2):
	return (
		np.logical_and(arr1, arr2).sum() 
		/ float(np.logical_or(arr1, arr2).sum())
	)


def get_jaccards(decisions, times, feat_gt):
	ret_times = [0]
	useful_jaccards = [0.0]
	useless_jaccards = [0.0]

	for dec, time in zip(decisions, times):
		ret_times.append(time)
		useful_jaccards.append(get_bool_jaccard(feat_gt, dec == 1))
		useless_jaccards.append(get_bool_jaccard(feat_gt == False, dec == -1))

	return useful_jaccards, useless_jaccards, ret_times


def to_stochastic(selector_tuple):
	""" converts gbt selectors to sgbt """
	new_dict = dict()
	new_dict['name'] = 'stoch_' + selector_tuple[0]['name']
	new_dict['batch'] = selector_tuple[0]['batch']
	new_dict['underlying_model'] = 'Stochastic Gradient Boosted Trees'
	new_dict['model_class'] = 'Boruta Stochastic Gradient Boosted Trees'

	new_mod = clone(selector_tuple[1])
	new_mod.estimator.subsample = .5

	return (new_dict, new_mod)


if __name__ == '__main__':

	SAVE_DIR = 'run_res'
	TESTS = ['hc2', 'madelon4', 'madelon2']
	ENVIRONMENT = 'laptop'
	
	# rand_seed = np.random.randint(1e9)
	rand_seed = None
	
	selectors_rf = [
		(	{
				'name': 'rand_forest_auto',
				'batch': False,
				'underlying_model': 'Random Forest',
				'model_class': 'Boruta Random Forest'
			},
			TrackingBorutaPy(
				RandomForestClassifier(verbose=0, n_jobs=-1),
				n_estimators='auto',
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'rand_forest_auto_max_depth_5',
				'batch': False,
				'underlying_model': 'Random Forest',
				'model_class': 'Boruta Random Forest'
			},
			TrackingBorutaPy(
				RandomForestClassifier(verbose=0, max_depth=5, n_jobs=-1),
				n_estimators='auto',
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'rand_forest_100_max_depth_5',
				'batch': False,
				'underlying_model': 'Random Forest',
				'model_class': 'Boruta Random Forest'
			},
			TrackingBorutaPy(
				RandomForestClassifier(verbose=0, max_depth=5, n_jobs=-1),
				n_estimators=100,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'rand_forest_100_max_depth_7',
				'batch': False,
				'underlying_model': 'Random Forest',
				'model_class': 'Boruta Random Forest'
			},
			TrackingBorutaPy(
				RandomForestClassifier(verbose=0, max_depth=7, n_jobs=-1),
				n_estimators=100,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'rand_forest_1000_max_depth_7',
				'batch': False,
				'underlying_model': 'Random Forest',
				'model_class': 'Boruta Random Forest'
			},
			TrackingBorutaPy(
				RandomForestClassifier(verbose=0, max_depth=7, n_jobs=-1),
				n_estimators=1000,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		)
	]

	selectors_rf_old = [
		(	{
				'name': 'rand_forest_auto',
				'batch': False,
				'underlying_model': 'Random Forest',
				'model_class': 'Boruta Random Forest'
			},
			TrackingBorutaPy(
				RandomForestClassifier(verbose=0, n_jobs=-1),
				n_estimators='auto',
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'rand_forest_auto_max_depth_3',
				'batch': False,
				'underlying_model': 'Random Forest',
				'model_class': 'Boruta Random Forest'
			},
			TrackingBorutaPy(
				RandomForestClassifier(verbose=0, max_depth=3, n_jobs=-1),
				n_estimators='auto',
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'rand_forest_auto_max_depth_5',
				'batch': False,
				'underlying_model': 'Random Forest',
				'model_class': 'Boruta Random Forest'
			},
			TrackingBorutaPy(
				RandomForestClassifier(verbose=0, max_depth=5, n_jobs=-1),
				n_estimators='auto',
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'rand_forest_auto_max_depth_7',
				'batch': False,
				'underlying_model': 'Random Forest',
				'model_class': 'Boruta Random Forest'
			},
			TrackingBorutaPy(
				RandomForestClassifier(verbose=0, max_depth=7, n_jobs=-1),
				n_estimators='auto',
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'rand_forest_100',
				'batch': False,
				'underlying_model': 'Random Forest',
				'model_class': 'Boruta Random Forest'
			},
			TrackingBorutaPy(
				RandomForestClassifier(verbose=0, n_jobs=-1),
				n_estimators=100,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'rand_forest_100_max_depth_3',
				'batch': False,
				'underlying_model': 'Random Forest',
				'model_class': 'Boruta Random Forest'
			},
			TrackingBorutaPy(
				RandomForestClassifier(verbose=0, max_depth=3, n_jobs=-1),
				n_estimators=100,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'rand_forest_100_max_depth_5',
				'batch': False,
				'underlying_model': 'Random Forest',
				'model_class': 'Boruta Random Forest'
			},
			TrackingBorutaPy(
				RandomForestClassifier(verbose=0, max_depth=5, n_jobs=-1),
				n_estimators=100,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'rand_forest_100_max_depth_7',
				'batch': False,
				'underlying_model': 'Random Forest',
				'model_class': 'Boruta Random Forest'
			},
			TrackingBorutaPy(
				RandomForestClassifier(verbose=0, max_depth=7, n_jobs=-1),
				n_estimators=100,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'rand_forest_1000',
				'batch': False,
				'underlying_model': 'Random Forest',
				'model_class': 'Boruta Random Forest'
			},
			TrackingBorutaPy(
				RandomForestClassifier(verbose=0, n_jobs=-1),
				n_estimators=1000,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'rand_forest_1000_max_depth_3',
				'batch': False,
				'underlying_model': 'Random Forest',
				'model_class': 'Boruta Random Forest'
			},
			TrackingBorutaPy(
				RandomForestClassifier(verbose=0, max_depth=3, n_jobs=-1),
				n_estimators=1000,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'rand_forest_1000_max_depth_5',
				'batch': False,
				'underlying_model': 'Random Forest',
				'model_class': 'Boruta Random Forest'
			},
			TrackingBorutaPy(
				RandomForestClassifier(verbose=0, max_depth=5, n_jobs=-1),
				n_estimators=1000,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'rand_forest_1000_max_depth_7',
				'batch': False,
				'underlying_model': 'Random Forest',
				'model_class': 'Boruta Random Forest'
			},
			TrackingBorutaPy(
				RandomForestClassifier(verbose=0, max_depth=7, n_jobs=-1),
				n_estimators=1000,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		)
	]

	selectors_et = [
		(	{
				'name': 'extra_trees_auto',
				'batch': False,
				'underlying_model': 'Extra Trees',
				'model_class': 'Boruta Extra Trees'
			},
			TrackingBorutaPy(
				ExtraTreesClassifier(verbose=0, n_jobs=-1),
				n_estimators='auto',
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'extra_trees_auto_max_depth_7',
				'batch': False,
				'underlying_model': 'Extra Trees',
				'model_class': 'Boruta Extra Trees'
			},
			TrackingBorutaPy(
				ExtraTreesClassifier(verbose=0, max_depth=7, n_jobs=-1),
				n_estimators='auto',
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'extra_trees_100',
				'batch': False,
				'underlying_model': 'Extra Trees',
				'model_class': 'Boruta Extra Trees'
			},
			TrackingBorutaPy(
				ExtraTreesClassifier(verbose=0, n_jobs=-1),
				n_estimators=100,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'extra_trees_100_max_depth_7',
				'batch': False,
				'underlying_model': 'Extra Trees',
				'model_class': 'Boruta Extra Trees'
			},
			TrackingBorutaPy(
				ExtraTreesClassifier(verbose=0, max_depth=7, n_jobs=-1),
				n_estimators=100,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'extra_trees_1000',
				'batch': False,
				'underlying_model': 'Extra Trees',
				'model_class': 'Boruta Extra Trees'
			},
			TrackingBorutaPy(
				ExtraTreesClassifier(verbose=0, n_jobs=-1),
				n_estimators=1000,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		)
	]

	selectors_et_old = [
		(	{
				'name': 'extra_trees_auto',
				'batch': False,
				'underlying_model': 'Extra Trees',
				'model_class': 'Boruta Extra Trees'
			},
			TrackingBorutaPy(
				ExtraTreesClassifier(verbose=0, n_jobs=-1),
				n_estimators='auto',
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'extra_trees_auto_max_depth_3',
				'batch': False,
				'underlying_model': 'Extra Trees',
				'model_class': 'Boruta Extra Trees'
			},
			TrackingBorutaPy(
				ExtraTreesClassifier(verbose=0, max_depth=3, n_jobs=-1),
				n_estimators='auto',
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'extra_trees_auto_max_depth_5',
				'batch': False,
				'underlying_model': 'Extra Trees',
				'model_class': 'Boruta Extra Trees'
			},
			TrackingBorutaPy(
				ExtraTreesClassifier(verbose=0, max_depth=5, n_jobs=-1),
				n_estimators='auto',
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'extra_trees_auto_max_depth_7',
				'batch': False,
				'underlying_model': 'Extra Trees',
				'model_class': 'Boruta Extra Trees'
			},
			TrackingBorutaPy(
				ExtraTreesClassifier(verbose=0, max_depth=7, n_jobs=-1),
				n_estimators='auto',
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'extra_trees_100',
				'batch': False,
				'underlying_model': 'Extra Trees',
				'model_class': 'Boruta Extra Trees'
			},
			TrackingBorutaPy(
				ExtraTreesClassifier(verbose=0, n_jobs=-1),
				n_estimators=100,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'extra_trees_100_max_depth_3',
				'batch': False,
				'underlying_model': 'Extra Trees',
				'model_class': 'Boruta Extra Trees'
			},
			TrackingBorutaPy(
				ExtraTreesClassifier(verbose=0, max_depth=3, n_jobs=-1),
				n_estimators=100,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'extra_trees_100_max_depth_5',
				'batch': False,
				'underlying_model': 'Extra Trees',
				'model_class': 'Boruta Extra Trees'
			},
			TrackingBorutaPy(
				ExtraTreesClassifier(verbose=0, max_depth=5, n_jobs=-1),
				n_estimators=100,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'extra_trees_100_max_depth_7',
				'batch': False,
				'underlying_model': 'Extra Trees',
				'model_class': 'Boruta Extra Trees'
			},
			TrackingBorutaPy(
				ExtraTreesClassifier(verbose=0, max_depth=7, n_jobs=-1),
				n_estimators=100,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'extra_trees_1000',
				'batch': False,
				'underlying_model': 'Extra Trees',
				'model_class': 'Boruta Extra Trees'
			},
			TrackingBorutaPy(
				ExtraTreesClassifier(verbose=0, n_jobs=-1),
				n_estimators=1000,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'extra_trees_1000_max_depth_3',
				'batch': False,
				'underlying_model': 'Extra Trees',
				'model_class': 'Boruta Extra Trees'
			},
			TrackingBorutaPy(
				ExtraTreesClassifier(verbose=0, max_depth=3, n_jobs=-1),
				n_estimators=1000,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'extra_trees_1000_max_depth_5',
				'batch': False,
				'underlying_model': 'Extra Trees',
				'model_class': 'Boruta Extra Trees'
			},
			TrackingBorutaPy(
				ExtraTreesClassifier(verbose=0, max_depth=5, n_jobs=-1),
				n_estimators=1000,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'extra_trees_1000_max_depth_7',
				'batch': False,
				'underlying_model': 'Extra Trees',
				'model_class': 'Boruta Extra Trees'
			},
			TrackingBorutaPy(
				ExtraTreesClassifier(verbose=0, max_depth=7, n_jobs=-1),
				n_estimators=1000,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		)
	]

	selectors_gb = [
		(	{
				'name': 'grad_boost_auto_max_depth_3_lr_25',
				'batch': False,
				'underlying_model': 'Gradient Boosted Trees',
				'model_class': 'Boruta Gradient Boosted Trees'
			},
			TrackingBorutaPy(
				GradientBoostingClassifier(verbose=0, max_depth=3, learning_rate=.25),
				n_estimators='auto',
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'grad_boost_auto_max_depth_3_lr_1',
				'batch': False,
				'underlying_model': 'Gradient Boosted Trees',
				'model_class': 'Boruta Gradient Boosted Trees'
			},
			TrackingBorutaPy(
				GradientBoostingClassifier(verbose=0, max_depth=3, learning_rate=.1),
				n_estimators='auto',
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'grad_boost_auto_max_depth_5_lr_25',
				'batch': False,
				'underlying_model': 'Gradient Boosted Trees',
				'model_class': 'Boruta Gradient Boosted Trees'
			},
			TrackingBorutaPy(
				GradientBoostingClassifier(verbose=0, max_depth=5, learning_rate=.25),
				n_estimators='auto',
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'grad_boost_auto_max_depth_5_lr_1',
				'batch': False,
				'underlying_model': 'Gradient Boosted Trees',
				'model_class': 'Boruta Gradient Boosted Trees'
			},
			TrackingBorutaPy(
				GradientBoostingClassifier(verbose=0, max_depth=5, learning_rate=.1),
				n_estimators='auto',
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'grad_boost_100_max_depth_3_lr_25',
				'batch': False,
				'underlying_model': 'Gradient Boosted Trees',
				'model_class': 'Boruta Gradient Boosted Trees'
			},
			TrackingBorutaPy(
				GradientBoostingClassifier(verbose=0, max_depth=3, learning_rate=.25),
				n_estimators=100,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'grad_boost_100_max_depth_3_lr_1',
				'batch': False,
				'underlying_model': 'Gradient Boosted Trees',
				'model_class': 'Boruta Gradient Boosted Trees'
			},
			TrackingBorutaPy(
				GradientBoostingClassifier(verbose=0, max_depth=3, learning_rate=.1),
				n_estimators=100,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'grad_boost_100_max_depth_5_lr_25',
				'batch': False,
				'underlying_model': 'Gradient Boosted Trees',
				'model_class': 'Boruta Gradient Boosted Trees'
			},
			TrackingBorutaPy(
				GradientBoostingClassifier(verbose=0, max_depth=5, learning_rate=.25),
				n_estimators=100,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'grad_boost_100_max_depth_5_lr_1',
				'batch': False,
				'underlying_model': 'Gradient Boosted Trees',
				'model_class': 'Boruta Gradient Boosted Trees'
			},
			TrackingBorutaPy(
				GradientBoostingClassifier(verbose=0, max_depth=5, learning_rate=.1),
				n_estimators=100,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'grad_boost_500_max_depth_3_lr_25',
				'batch': False,
				'underlying_model': 'Gradient Boosted Trees',
				'model_class': 'Boruta Gradient Boosted Trees'
			},
			TrackingBorutaPy(
				GradientBoostingClassifier(verbose=0, max_depth=3, learning_rate=.25),
				n_estimators=500,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'grad_boost_500_max_depth_3_lr_1',
				'batch': False,
				'underlying_model': 'Gradient Boosted Trees',
				'model_class': 'Boruta Gradient Boosted Trees'
			},
			TrackingBorutaPy(
				GradientBoostingClassifier(verbose=0, max_depth=3, learning_rate=.1),
				n_estimators=500,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'grad_boost_500_max_depth_5_lr_25',
				'batch': False,
				'underlying_model': 'Gradient Boosted Trees',
				'model_class': 'Boruta Gradient Boosted Trees'
			},
			TrackingBorutaPy(
				GradientBoostingClassifier(verbose=0, max_depth=5, learning_rate=.25),
				n_estimators=500,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		),
		(	{
				'name': 'grad_boost_500_max_depth_5_lr_1',
				'batch': False,
				'underlying_model': 'Gradient Boosted Trees',
				'model_class': 'Boruta Gradient Boosted Trees'
			},
			TrackingBorutaPy(
				GradientBoostingClassifier(verbose=0, max_depth=5, learning_rate=.1),
				n_estimators=500,
				two_step=False,
				random_state=rand_seed,
				max_iter=256,
				verbose=1
			)
		)
	]

	selectors_sgb = [to_stochastic(sel) for sel in selectors_gb]

	selectors_bb = [
		(	{
				'name': 'batch_rand_forest_auto',
				'batch': True,
				'underlying_model': 'Random Forest',
				'model_class': 'Batch Boruta Random Forest'
			},
			TrackingBatchBorutaPy(
				[
					RandomForestClassifier(verbose=0, n_jobs=-1),
					RandomForestClassifier(verbose=0, max_depth=3, n_jobs=-1),
					RandomForestClassifier(verbose=0, max_depth=5, n_jobs=-1), 
					RandomForestClassifier(verbose=0, max_depth=7, n_jobs=-1)
				], 
				n_estimators='auto',
				two_step=False,
				max_iter=150,
				mode='mp',
				n_jobs=-1,
				verbose=1,
				random_state=rand_seed
			)
		),
		(	{
				'name': 'batch_rand_forest_100',
				'batch': True,
				'underlying_model': 'Random Forest',
				'model_class': 'Batch Boruta Random Forest'
			},
			TrackingBatchBorutaPy(
				[
					RandomForestClassifier(verbose=0, n_jobs=-1),
					RandomForestClassifier(verbose=0, max_depth=3, n_jobs=-1),
					RandomForestClassifier(verbose=0, max_depth=5, n_jobs=-1), 
					RandomForestClassifier(verbose=0, max_depth=7, n_jobs=-1)
				], 
				n_estimators=[100]*4,
				two_step=False,
				max_iter=150,
				mode='mp',
				n_jobs=-1,
				verbose=1,
				random_state=rand_seed
			)
		),
		(	{
				'name': 'batch_rand_forest_1000',
				'batch': True,
				'underlying_model': 'Random Forest',
				'model_class': 'Batch Boruta Random Forest'
			},
			TrackingBatchBorutaPy(
				[
					RandomForestClassifier(verbose=0, n_jobs=-1),
					RandomForestClassifier(verbose=0, max_depth=3, n_jobs=-1),
					RandomForestClassifier(verbose=0, max_depth=5, n_jobs=-1), 
					RandomForestClassifier(verbose=0, max_depth=7, n_jobs=-1)
				], 
				n_estimators=[1000]*4,
				two_step=False,
				max_iter=150,
				mode='mp',
				n_jobs=-1,
				verbose=1,
				random_state=rand_seed
			)
		),
		(	{
				'name': 'batch_extra_trees_auto',
				'batch': True,
				'underlying_model': 'Extra Trees',
				'model_class': 'Batch Boruta Extra Trees'
			},
			TrackingBatchBorutaPy(
				[
					ExtraTreesClassifier(verbose=0, n_jobs=-1),
					ExtraTreesClassifier(verbose=0, max_depth=3, n_jobs=-1),
					ExtraTreesClassifier(verbose=0, max_depth=5, n_jobs=-1), 
					ExtraTreesClassifier(verbose=0, max_depth=7, n_jobs=-1)
				], 
				n_estimators='auto',
				two_step=False,
				max_iter=150,
				mode='mp',
				n_jobs=-1,
				verbose=1,
				random_state=rand_seed
			)
		),
		(	{
				'name': 'batch_extra_trees_100',
				'batch': True,
				'underlying_model': 'Extra Trees',
				'model_class': 'Batch Boruta Extra Trees'
			},
			TrackingBatchBorutaPy(
				[
					ExtraTreesClassifier(verbose=0, n_jobs=-1),
					ExtraTreesClassifier(verbose=0, max_depth=3, n_jobs=-1),
					ExtraTreesClassifier(verbose=0, max_depth=5, n_jobs=-1), 
					ExtraTreesClassifier(verbose=0, max_depth=7, n_jobs=-1)
				], 
				n_estimators=[100]*4,
				two_step=False,
				max_iter=150,
				mode='mp',
				n_jobs=-1,
				verbose=1,
				random_state=rand_seed
			)
		),
		(	{
				'name': 'batch_extra_trees_1000',
				'batch': True,
				'underlying_model': 'Extra Trees',
				'model_class': 'Batch Boruta Extra Trees'
			},
			TrackingBatchBorutaPy(
				[
					ExtraTreesClassifier(verbose=0, n_jobs=-1),
					ExtraTreesClassifier(verbose=0, max_depth=3, n_jobs=-1),
					ExtraTreesClassifier(verbose=0, max_depth=5, n_jobs=-1), 
					ExtraTreesClassifier(verbose=0, max_depth=7, n_jobs=-1)
				], 
				n_estimators=[1000]*4,
				two_step=False,
				max_iter=150,
				mode='mp',
				n_jobs=-1,
				verbose=1,
				random_state=rand_seed
			)
		),
		(	{
				'name': 'batch_mix_nonauto',
				'batch': True,
				'underlying_model': 'Both',
				'model_class': 'Batch Boruta Mix'
			},
			TrackingBatchBorutaPy(
				[
					RandomForestClassifier(verbose=0, max_depth=5, n_jobs=-1), 
					ExtraTreesClassifier(verbose=0, n_jobs=-1),
					RandomForestClassifier(verbose=0, max_depth=7, n_jobs=-1),
					ExtraTreesClassifier(verbose=0, n_jobs=-1),
					RandomForestClassifier(verbose=0, max_depth=7, n_jobs=-1),
					ExtraTreesClassifier(verbose=0, max_depth=7, n_jobs=-1),
				], 
				n_estimators=[100, 1000, 100, 100, 1000, 100],
				two_step=False,
				max_iter=150,
				mode='mp',
				n_jobs=-1,
				verbose=1,
				random_state=rand_seed
			)
		),
		(	{
				'name': 'batch_mix_auto',
				'batch': True,
				'underlying_model': 'Both',
				'model_class': 'Batch Boruta Mix'
			},
			TrackingBatchBorutaPy(
				[
					RandomForestClassifier(verbose=0, n_jobs=-1),
					RandomForestClassifier(verbose=0, max_depth=7, n_jobs=-1),
					RandomForestClassifier(verbose=0, max_depth=5, n_jobs=-1),
					ExtraTreesClassifier(verbose=0, max_depth=7, n_jobs=-1),
					ExtraTreesClassifier(verbose=0, n_jobs=-1),
				], 
				n_estimators='auto',
				two_step=False,
				max_iter=150,
				mode='mp',
				n_jobs=-1,
				verbose=1,
				random_state=rand_seed
			)
		),
	]

	selectors = selectors_bb

	for test in TESTS:

		print("CURRENT TEST:\t{}".format(test))

		X, y, feat_gt = get_test_data(test)

		for selector_data, selector in selectors:

			print("CURRENT SELECTOR:\n{}".format(selector_data))

			hist = selector.fit(X, y)
			finish_time = int(time.time())

			jaccard = jaccard_score(feat_gt, selector.support_, pos_label=True)
			precision = precision_score(feat_gt, selector.support_, pos_label=True)
			recall = recall_score(feat_gt, selector.support_, pos_label=True)
			f1 = f1_score(feat_gt, selector.support_, pos_label=True)
			comp = completeness_score(feat_gt, selector.support_)
			
			useful_jaccards, useless_jaccards, times = get_jaccards(*hist, feat_gt)

			df = pd.DataFrame(
				list(zip(times, useful_jaccards, useless_jaccards)),
				columns=['time', 'jaccard_useful', 'jaccard_useless']
			)

			df['model_name'] = selector_data['name']
			df['test'] = test
			df['env'] = ENVIRONMENT
			df['batch'] = selector_data['batch']
			df['underlying_model'] = selector_data['underlying_model']
			df['model_class'] = selector_data['model_class']
			df['finish_time'] = finish_time
			df['jaccard'] = jaccard
			df['precision'] = precision
			df['recall'] = recall
			df['f1'] = f1
			df['completeness'] = comp

			df.to_csv(
				os.path.join(
					SAVE_DIR,
					'{}__{}__{}__{}.csv'.format(
						selector_data['name'],
						test,
						ENVIRONMENT,
						finish_time
					)
				),
				index=False
			)