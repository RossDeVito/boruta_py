import numpy as np
import pandas as pd 
from sklearn.datasets import make_classification

data = make_classification(
			n_samples=2000,
			n_features=500,
			n_informative=5,
			n_redundant=5,
			n_repeated=10,
			class_sep=2.0,
			shuffle=False,
			random_state=147
		)

feat_gt = np.asarray([*([True] * 20), *([False] * 480)])

# pickled tuple: (X, y, feat_ground_truth)
pd.to_pickle((data[0], data[1], feat_gt), 'madelon.pkl')

data = make_classification(
			n_samples=2000,
			n_features=500,
			n_clusters_per_class=16,
			n_informative=5,
			n_redundant=5,
			n_repeated=10,
			class_sep=2.0,
			shuffle=False,
			random_state=147
		)

feat_gt = np.asarray([*([True] * 20), *([False] * 480)])

# pickled tuple: (X, y, feat_ground_truth)
pd.to_pickle((data[0], data[1], feat_gt), 'madelon2.pkl')

data = make_classification(
			n_samples=2000,
			n_features=500,
			n_informative=5,
			n_redundant=5,
			n_repeated=10,
			class_sep=1.0,
			shuffle=False,
			random_state=147
		)

feat_gt = np.asarray([*([True] * 20), *([False] * 480)])

# pickled tuple: (X, y, feat_ground_truth)
pd.to_pickle((data[0], data[1], feat_gt), 'madelon3.pkl')

data = make_classification(
			n_samples=2000,
			n_features=500,
			n_clusters_per_class=16,
			n_informative=5,
			n_redundant=5,
			n_repeated=10,
			class_sep=1.0,
			shuffle=False,
			random_state=147
		)

feat_gt = np.asarray([*([True] * 20), *([False] * 480)])

# pickled tuple: (X, y, feat_ground_truth)
pd.to_pickle((data[0], data[1], feat_gt), 'madelon4.pkl')