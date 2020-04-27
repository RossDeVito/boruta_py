import numpy as np
import pandas as pd 
from sklearn.datasets import make_classification, make_moons
from sklearn.datasets import make_gaussian_quantiles

data = make_classification(
			n_samples=3000,
			n_features=200,
			n_informative=40,
			n_redundant=10,
			n_repeated=5,
			class_sep=1.0,
			shuffle=False,
			flip_y=.02,
			random_state=147
		)

feat_gt = np.asarray([*([True] * 55), *([False] * 145)])
pd.to_pickle((data[0], data[1], feat_gt), 'hc_bl.pkl')

# MOON
moon_data = make_moons(n_samples=4000)

feat_gt = np.asarray([*([True] * 2), *([False] * 0)])

# pickled tuple: (X, y, feat_ground_truth)
pd.to_pickle((moon_data[0], moon_data[1], feat_gt), 'hcm_1.pkl')

moon_data = make_moons(n_samples=3000, noise=0.32)
moon_data = (
	np.hstack((
		moon_data[0], 
		moon_data[0] * [np.random.rand(), np.random.randint(1e9)],
		np.vstack(moon_data[0][:, 0] * np.random.rand())
	)),
	moon_data[1]
)

feat_gt = np.asarray([*([True] * 5), *([False] * 0)])

# pickled tuple: (X, y, feat_ground_truth)
pd.to_pickle((moon_data[0], moon_data[1], feat_gt), 'hcm_2.pkl')

g1 = make_gaussian_quantiles(cov=3., n_samples=1500, n_features=10, n_classes=2)
g2 = make_gaussian_quantiles(
	mean=(4, 4, 4, 4, 4, 4, 4, 3, 4, 4), n_samples=1500, n_features=10, n_classes=2)

gc = (np.concatenate((g1[0], g2[0])), np.concatenate((g1[1], - g2[1] + 1)))

feat_gt = np.asarray([*([True] * 10), *([False] * 0)])

# pickled tuple: (X, y, feat_ground_truth)
pd.to_pickle((gc[0], gc[1], feat_gt), 'gc.pkl')

xor_y = np.bitwise_xor(moon_data[1], gc[1])
print('xor sum: ', xor_y.sum())

y = data[1] + xor_y

X = np.hstack((data[0], moon_data[0], gc[0]))

feat_gt = np.asarray([*([True] * 55), *([False] * 145), *([True] * 15)])

pd.to_pickle((X, y, feat_gt), 'hc2.pkl')