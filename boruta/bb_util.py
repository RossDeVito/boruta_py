import os

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import iplot


def get_test_data(test):
	if test == 'madelon':
		test_data = pd.read_pickle(
			os.path.join('..', 'data', 'madelon.pkl')
		)
		return test_data
	elif test == 'madelon2':
		test_data = pd.read_pickle(
			os.path.join('..', 'data', 'madelon2.pkl')
		)
		return test_data
	elif test == 'madelon3':
		test_data = pd.read_pickle(
			os.path.join('..', 'data', 'madelon3.pkl')
		)
		return test_data
	elif test == 'madelon4':
		test_data = pd.read_pickle(
			os.path.join('..', 'data', 'madelon4.pkl')
		)
		return test_data
	elif test == 'hcm_1':
		test_data = pd.read_pickle(
			os.path.join('..', 'data', 'hcm_1.pkl')
		)
		return test_data
	elif test == 'hcm_2':
		test_data = pd.read_pickle(
			os.path.join('..', 'data', 'hcm_2.pkl')
		)
		return test_data
	elif test == 'gc':
		test_data = pd.read_pickle(
			os.path.join('..', 'data', 'gc.pkl')
		)
		return test_data
	elif test == 'hc_bl':
		test_data = pd.read_pickle(
			os.path.join('..', 'data', 'hc_bl.pkl')
		)
		return test_data
	elif test == 'hc2':
		test_data = pd.read_pickle(
			os.path.join('..', 'data', 'hc2.pkl')
		)
		return test_data


# from https://github.com/faizanahemad/data-science/blob/master/exploration_projects/imbalance-noise-oversampling/lib.py
def visualize_2d(X,y,algorithm="tsne",title="Data in 2D",figsize=(8,8)):
	from sklearn.manifold import TSNE
	from sklearn.decomposition import PCA
	
	if algorithm=="tsne":
		reducer = TSNE(n_components=2,random_state=47,n_iter=400,angle=0.6)
	elif algorithm=="pca":
		reducer = PCA(n_components=2,random_state=47)
	else:
		raise ValueError("Unsupported dimensionality reduction algorithm given.")
	if X.shape[1]>2:
		X = reducer.fit_transform(X)
	else:
		if type(X)==pd.DataFrame:
			X=X.values
	f, (ax1) = plt.subplots(nrows=1, ncols=1,figsize=figsize)
	sns.scatterplot(X[:,0],X[:,1],hue=y,ax=ax1)
	ax1.set_title(title)
	plt.show()


def visualize_2d_quad(X, y, real_feat_mask, title="Data in 2D", figsize=(8,8),
						alpha=.6):
	from sklearn.manifold import TSNE
	from sklearn.decomposition import PCA

	y = pd.Series(y.astype(str)).astype('category')
	
	reducer_tsne = TSNE(n_components=2,random_state=147,n_iter=400,angle=0.6)
	reducer_pca = PCA(n_components=2,random_state=147)

	f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=figsize)
	
	X_tsne = reducer_tsne.fit_transform(X)
	sns.scatterplot(X_tsne[:,0], X_tsne[:,1], hue=y, ax=ax1, alpha=alpha)
	ax1.set_title("TSNE 2D (All Feats)")

	X_pca = reducer_pca.fit_transform(X)
	sns.scatterplot(X_pca[:,0], X_pca[:,1], hue=y, ax=ax2, alpha=alpha)
	ax2.set_title("PCA 2D (All Feats)")

	X_tsne = reducer_tsne.fit_transform(X[:, real_feat_mask])
	sns.scatterplot(X_tsne[:,0], X_tsne[:,1], hue=y, ax=ax3, alpha=alpha)
	ax3.set_title("TSNE 2D (Real Feats Only)")

	X_pca = reducer_pca.fit_transform(X[:, real_feat_mask])
	sns.scatterplot(X_pca[:,0], X_pca[:,1], hue=y, ax=ax4, alpha=alpha)
	ax4.set_title("PCA 2D (Real Feats Only)")
	
	f.tight_layout(rect=[0, 0.03, 1, 0.95])
	f.suptitle(title, fontsize=20)

	ax = f.get_axes()
	
	plt.show()


def visualize_3d(X,y,algorithm="tsne",title="Data in 3D"):
	from sklearn.manifold import TSNE
	from sklearn.decomposition import PCA
	
	if algorithm=="tsne":
		reducer = TSNE(n_components=3,random_state=47,n_iter=400,angle=0.6)
	elif algorithm=="pca":
		reducer = PCA(n_components=3,random_state=47)
	else:
		raise ValueError("Unsupported dimensionality reduction algorithm given.")
	
	if X.shape[1]>3:
		X = reducer.fit_transform(X)
	else:
		if type(X)==pd.DataFrame:
			X=X.values
	
	marker_shapes = ["circle","diamond", "circle-open", "square",  "diamond-open", "cross","square-open",]
	traces = []
	for hue in np.unique(y):
		X1 = X[y==hue]

		trace = go.Scatter3d(
			x=X1[:,0],
			y=X1[:,1],
			z=X1[:,2],
			mode='markers',
			name = str(hue),
			marker=dict(
				size=12,
				symbol=marker_shapes.pop(),
				line=dict(
					width=int(np.random.randint(3,10)/10)
				),
				opacity=int(np.random.randint(6,10)/10)
			)
		)
		traces.append(trace)


	layout = go.Layout(
		title=title,
		scene=dict(
			xaxis=dict(
				title='Dim 1'),
			yaxis=dict(
				title='Dim 2'),
			zaxis=dict(
				title='Dim 3'), ),
		margin=dict(
			l=0,
			r=0,
			b=0,
			t=0
		)
	)
	fig = go.Figure(data=traces, layout=layout)
	iplot(fig)
