import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import time
from mpl_toolkits.mplot3d import Axes3D
import sys
from helper import *


def converge(j1, j2):
	# print(j1, j2)
	if abs(j1 - j2) < 1e-15:
		return True
	else:
		return False
def cost(X, Y, theta):
	'''Input:
		X = Design Matrix 
		# h= vector of hypothesis for each exampla x_i (i=0...m) (Vector of Xθ)
		Y= Output vector
        theta = parameter matrix
            [θ0]
            [θ1]
            [θ2]
		Returns the cost J(θ)
		J(θ) = 1/2m * Σ(hθ(x_i) - y_i)^2 (i=0...m)
		or
		J(θ) = 1/2m * [(Xθ - Y)^t * (Xθ - Y)]
	'''
	m = len(Y)
	h = np.dot(X, theta)

	return (1 / ((2 * m)) * np.sum((h - Y) ** 2))

def gradientDescent(X, Y, eta):
	'''
	Input:
		X= Design Matrix of x
		Y= Output
		eta = Learning rate

	output: θ vector
		θ := θ - eta * (1/m) * X^t * (Xθ - Y)
	'''

	cost_arr = []
	theta0_arr = []
	theta1_arr = []
	m = len(Y)
	theta_prev = np.zeros((X.shape[1], 1))
	cost_prev = 100.0
	cost_curr = cost(X, Y, theta_prev)

	# print(type(cost_prev), type(cost_curr))

	itr_count = 0
	# for i in range(0, 100000):
	while not converge(cost_prev, cost_curr):
		itr_count+=1
		cost_prev = cost_curr
		cost_arr.append(cost_curr)
		theta0_arr.append(theta_prev[0][0])
		theta1_arr.append(theta_prev[1][0])
		h = np.dot(X, theta_prev)

		theta_current = theta_prev - ((1 / m) * eta) *  np.dot(np.transpose(X), (h - Y))
		theta_prev = theta_current
		cost_curr = cost(X, Y, theta_prev)

        

	return {"theta": theta_current, "cost_arr": cost_arr, "theta0_arr": theta0_arr, "theta1_arr": theta1_arr, "itr_count": itr_count, "cost": cost_curr}



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



def loadData(x_file, y_file):
	'''Load data linearX.csv and linearY.csv
		Return type: dict
		{'X': X, 'Y' : Y}
	'''
	X = designMatrix(normalize(np.array(pd.read_csv(x_file, header=None))))
	Y = np.array(pd.read_csv(y_file, header=None))

	return {'X' : X, 'Y': Y}



def partC(X, Y, theta_opt, cost_arr, theta0_arr, theta1_arr):
	theta0_center = theta_opt[0][0]
	theta1_center = theta_opt[1][0]

	theta0_axis = np.linspace(theta0_center -1 , theta0_center + 1, 100)
	theta1_axis = np.linspace(theta1_center -1 , theta1_center + 1, 100)


	# print(np.array(cost_axis))
	theta0_axis = np.array(theta0_axis)
	theta1_axis = np.array(theta1_axis)

	meshX, meshY = np.meshgrid(theta0_axis, theta1_axis)

	cost_axis = []

	for e1 in range((meshX.shape[0])):
		v = []
		for e2 in range(meshY.shape[0]):
			theta_vec = np.array([[meshX[e1][e2]], [meshY[e1][e2]]])
			c = cost(X, Y, theta_vec)
			v.append(c)
		cost_axis.append(v)    

	fig = plt.figure()
	ax = plt.axes(projection='3d')

	ax.plot_surface(meshX, meshY, np.array(cost_axis),cmap='viridis', edgecolor='none', label="Cost", alpha=0.7)

	def animate(i, theta0_arr, theta1_arr, cost_arr):
		line, = plt.plot(theta0_arr[:i], theta1_arr[:i], cost_arr[:i], color="red", marker=".")
		return line,

	anim = animation.FuncAnimation(fig, animate, frames=len(cost_arr), fargs=(theta0_arr, theta1_arr, cost_arr), interval=200)
	# ax.legend()
	ax.set_zlabel("J(θ)", fontweight='bold')
	ax.set_xlabel("θ1", fontweight='bold')
	ax.set_ylabel("θ2", fontweight='bold')

	plt.title("3D cost function")
	plt.show()


def partD(X, Y, theta_opt, cost_arr, theta0_arr, theta1_arr):
	theta0_center = theta_opt[0][0]
	theta1_center = theta_opt[1][0]

	theta0_axis = np.linspace(theta0_center -1 , theta0_center + 1, 100)
	theta1_axis = np.linspace(theta1_center -1 , theta1_center + 1, 100)


	# print(np.array(cost_axis))
	theta0_axis = np.array(theta0_axis)
	theta1_axis = np.array(theta1_axis)

	meshX, meshY = np.meshgrid(theta0_axis, theta1_axis)

	cost_axis = []

	for e1 in range((meshX.shape[0])):
		v = []
		for e2 in range(meshY.shape[0]):
			theta_vec = np.array([[meshX[e1][e2]], [meshY[e1][e2]]])
			c = cost(X, Y, theta_vec)
			v.append(c)
		cost_axis.append(v) 
	fig3 = plt.figure()

	plt.title("Contour plot")
	plt.contour(meshX, meshY, np.array(cost_axis))

	plt.xlabel("θ1", fontweight='bold')
	plt.ylabel("θ2", fontweight='bold')
	# plt.plot(theta0_arr, theta1_arr, color="red", marker=".")
	def animate(i, theta0_arr, theta1_arr):
		line, = plt.plot(theta0_arr[:i], theta1_arr[:i], color="red", marker=".")
		return line,

	anim = animation.FuncAnimation(fig3, animate, frames=len(cost_arr), fargs=(theta0_arr, theta1_arr), interval=200)
	plt.show()



def main(x_file, y_file, eta):
	data = loadData(x_file, y_file)

	X = data['X']
	Y = data['Y']


	print("Running Gradient Descent")
	gd = gradientDescent(X, Y, eta)
	theta_opt = gd["theta"]
	cost_arr = gd["cost_arr"]
	theta0_arr = gd["theta0_arr"]
	theta1_arr = gd["theta1_arr"]
	itr_count = gd["itr_count"]
	cost_final = gd["cost"]

	#---------part(a)------------------
	print("-"*10)
	print("part(a)")
	print("-"*10)
	print(f"Learning Rate = {eta}")
	print(f"Optimal theta = \n {theta_opt}")
	print(f"Number of iterations = {itr_count}")
	print(f"Cost = {cost_final}")

	print("-"*30)
	#-------------------------------------


	#---------part(b)------------------
	print("-"*10)
	print("part(b)")
	print("Showing plot..")
	plt.title("Linear regression - Hypothesis")
	plt.xlabel("Acidity")
	plt.ylabel("Density")
	plt.scatter(X[:,1], Y, label="Wine")
	plt.plot(X[:,1], X.dot(theta_opt), color='r', label="Hypothesis")
	plt.legend(loc="upper right")
	plt.show()
	print("-"*30)
	#-------------------------------------


	#---------part(c)------------------
	print("-"*10)
	print("part(c)")
	partC(X, Y, theta_opt, cost_arr, theta0_arr, theta1_arr)
	print("-"*30)
	#-------------------------------------
	
	#---------part(d)------------------
	print("-"*10)
	print("part(d)")
	partD(X, Y, theta_opt, cost_arr, theta0_arr, theta1_arr)
	print("-"*30)
	#-------------------------------------




if __name__ == "__main__":
	try:
	# x file, y file , eta

		x_file = sys.argv[1]
		y_file = sys.argv[2]
		eta = float(sys.argv[3])

		main(x_file, y_file, eta)
	except Exception as e:
		print(e)
		msg = '''
=>python3 q1.py x_file y_file eta
		'''
		print(msg)

	





















