import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import time
from mpl_toolkits.mplot3d import Axes3D
import sys
from helper import *


def loadData(x_file, y_file):
	'''Load data linearX.csv and linearY.csv
		Return type: dict
		{'X': X, 'Y' : Y}
	'''
	X = designMatrix(normalize(np.array(pd.read_csv(x_file, header=None))))
	Y = np.array(pd.read_csv(y_file, header=None))

	return {'X' : X, 'Y': Y}

def sigmoid(X, theta):
	return 1 / (1 + np.exp(-1 * np.dot(X, theta)))

def hessian(X, theta):
	sigm = sigmoid(X, theta)
	sigm1 = (1 - sigm)
	a = sigm * sigm1
	D = np.diag(a.T[0])
	return -1 *(X.T @ D @ X)

def deltaLL(X, Y, theta):
   return (X.T @ (Y - sigmoid(X, theta)))

def converge(theta1, theta2):
	if (abs(theta1 - theta2).all() < 1e-15):
		return True
	else:
		return False


def newton(X, Y):
	m = X.shape[1]
	theta_prev = np.ones((m,1))
	theta_current = np.zeros((m,1))
	itr_count = 0
	while(not converge(theta_current, theta_prev)):
		itr_count+=1
		theta_prev = theta_current.copy()
		H = hessian(X, theta_prev)
		d_ll_theta = deltaLL(X, Y, theta_prev)
		if np.linalg.det(H) == 0:
			theta = theta_prev - (np.linalg.pinv(H) @ d_ll_theta)
		else:
			theta = theta_prev - (np.linalg.inv(H) @ d_ll_theta)
		theta_current = theta.copy()
	return {"theta": theta_current, "itr_count": itr_count}


def getXData(X, Y, val = 0):
	m = X.shape[1]
	new_x = np.empty((0, m))
	for index, row in enumerate(Y):
		if row[0] == val:
			new_x = np.append(new_x, X[index])
	new_x = new_x.reshape((-1,m))   
	return new_x

def partA(X, Y):
	m = X.shape[1]
	print(m)
	theta = np.zeros((m,1))
	H = hessian(X, theta)
	theta_opt = newton(X, Y)
	return theta_opt

def partB(X, Y, theta_opt):
	xx0 = getXData(X, Y, val=0)
	x10 = xx0[:,1]
	x20 = xx0[:,2]

	xx1 = getXData(X, Y, val=1)
	x11 = xx1[:,1]
	x21 = xx1[:,2]

	plt.title("Logistic Regression")
	plt.xlabel("x1", fontweight="bold")
	plt.ylabel("x2", fontweight="bold")
	#todo filter x1, x2 wrt 0 and 1 y
	plt.scatter(x10, x20,color="r", label="Negative")
	plt.scatter(x11, x21,color="b", label="Positive")

	x_gen = np.linspace(-3, 3, 100)
	y_gen = -(theta_opt[0][0] + theta_opt[1][0]*x_gen) / theta_opt[2][0]
	yy = -(theta_opt[0][0] + theta_opt[1][0]*X[:,1]) / theta_opt[2][0]
	# plt.plot(X[:,1], yy,'-', linewidth=2, color="m", label="Decision Boundry")
	plt.plot(x_gen, y_gen,'-', linewidth=2, color="m", label="Decision Boundary")
	plt.legend(loc="upper right")
	plt.show()

def main(x_file, y_file):
	data = loadData(x_file, y_file)

	X = data['X']
	Y = data['Y']
	m = X.shape[1]

	newton_out = partA(X, Y)
	theta_opt = newton_out["theta"]
	itr_count = newton_out["itr_count"]

	#---------part(a)------------------
	print("-"*10)
	print("part(a)")
	print("-"*10)
	print(f"Optimal theta = \n {theta_opt}")
	print(f"Number of iteration = {itr_count}")
	print("-"*30)
	#-------------------------------------


	#---------part(b)------------------
	print("-"*10)
	print("part(b)")
	print("Showing plot..")
	print("-"*10)

	partB(X, Y, theta_opt)
	
	print("-"*30)
	#-------------------------------------



if __name__ == "__main__":
	try:
		# x file, y file 
		x_file = sys.argv[1]
		y_file = sys.argv[2]

		main(x_file, y_file)
	except Exception as e:
		print(e)
		msg = '''
=>python3 q3.py x_file y_file
		'''
		print(msg)




