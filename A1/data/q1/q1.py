import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from timing import time_it as tt
from mpl_toolkits.mplot3d import Axes3D
import sys



def loadData():
	'''Load data linearX.csv and linearY.csv
		Return type: dict
		{'X': X, 'Y' : Y}
	'''
	X = np.array(pd.read_csv('linearX.csv', header=None))
	Y = np.array(pd.read_csv('linearY.csv', header=None))

	return {'X' : X, 'Y': Y}


def normalize(V):
	'''Normalizes 1-D numpy vector
		(X - μ) / (max - min)
		or 
		(X - μ) / σ
	'''
	return (V - np.mean(V)) / np.std(V)



def hypothesis(theta, X):
	'''theta: Vector of parameters
		[ θ0 ]
		[ θ1 ]
		[ .. ] 
		[ θn ]

	X: Vector of features
		[ X0 ]
		[ X1 ]
		[ .. ] 
		[ Xn ]
	'''
	return np.dot(np.transpose(theta), X)


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




def cost(X, Y, theta):
	'''Input:
		X = Design Matrix 
		# h= vector of hypothesis for each exampla x_i (i=0...m) (Vector of Xθ)
		Y= Output vector
		Returns the cost J(θ)
		J(θ) = 1/2m * Σ(hθ(x_i) - y_i)^2 (i=0...m)
		or
		J(θ) = 1/2m * [(Xθ - Y)^t * (Xθ - Y)]
	'''
	m = len(Y)
	h = np.dot(X, theta)

	return (1 / ((2 * m)) * np.sum((h - Y) ** 2))



@tt
def gradientDescent(X, Y, eta):
	'''
	Input:
		X= Design Matrix of x
		Y= Output
		eta = Learning rate

	output: θ vector
		θ := θ - eta * (1/m) * X^t * (Xθ - Y)
	'''
	m = len(Y)
	theta_prev = np.zeros((X.shape[1], 1))
	for i in range(0, 10000):
		h = np.dot(X, theta_prev)
		theta_current = theta_prev - (1 / m) * eta *  np.dot(np.transpose(X), (h - Y))
		theta_prev = theta_current

	return theta_current

def main():
	data = loadData()
	X = data['X']
	Y = data['Y']

	desX = designMatrix(X)
	eta = 0.01
	theta = gradientDescent(desX, Y, 0.01)
	print(theta)
	print(cost(desX, Y, theta))

	plt.plot(cost_arr)


main()

# if __name__ == '__main__':
# 	# TODO: 
# 	# input x file, y file





