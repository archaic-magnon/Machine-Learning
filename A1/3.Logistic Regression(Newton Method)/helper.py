import numpy as np


def designMatrix(X):
	'''
	X:
		[x_1_0, x_1_0, ...x_i_0.. x_n_0 ]
		[x_1_1, x_1_1, ...x_i_1.. x_n_1 ]
		[       .......                 ]
		[x_1_m, x_1_m, ...x_i_m.. x_n_m ]

	return:
		[1 x_1_0, x_1_0, ...x_i_0.. x_n_0 ]
		[1 x_1_1, x_1_1, ...x_i_1.. x_n_1 ]
		[         .......                 ]
		[1 x_1_m, x_1_m, ...x_i_m.. x_n_m ]
	'''
	nrow = len(X)
	return np.append(np.ones((nrow, 1)), X, axis=1)


def normalize(V):
	'''Normalizes n-D numpy vector
		(X - mean) / (max - min)
	'''
	return (V - np.mean(V, axis=0)) / np.std(V, axis=0)
