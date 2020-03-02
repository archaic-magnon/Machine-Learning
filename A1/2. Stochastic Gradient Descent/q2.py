import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import time
from mpl_toolkits.mplot3d import Axes3D
import sys
from helper import *
import math


def loadData(test_file):
	test_data = designMatrix(np.array(pd.read_csv(test_file)))
	return {'test_data' : test_data}

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


def getNormalData(mean, sigma, count):
	return np.random.normal(mean, sigma, count).reshape(-1, 1)


def sample(count,theta):
	x0 = np.ones((count)).reshape(-1, 1)
	x1 = getNormalData(3, 2, count)
	x2 = getNormalData(-1, 2, count)
	# x2 = np.random.normal(-1, 2, count).reshape(-1, 1)
	error = getNormalData(0, math.sqrt(2), count)
	# error = np.random.normal(0, math.sqrt(2), count).reshape(-1, 1)
	x  = np.hstack((x0, x1, x2))
	y = np.dot(x, theta) + error

	xy = np.hstack((x, y))

	combined_data = np.hstack((x1, x2, y))

	np.random.shuffle(xy)

	x = xy[:,(0,1,2)]
	y = xy[:,3].reshape((-1, 1))

	return {"X":x, "Y":y, "C":combined_data}


def checkConverge(j1, j2):
	if (abs(j1 - j1) < 1e-15):
		return True
	else:
		return False


def SGD(X, Y, r):

	cost_arr = []
	theta0_arr = []
	theta1_arr = []
	theta2_arr = []

	epsilon = 1e-3

	if r == 1:
		epsilon = 1e-5
	if r == 100:
		epsilon = 1e-4
	if r == 10000:
		epsilon = 1e-6
	if r == 1000000:
		epsilon = 1e-9


	m = len(Y)
	itr_count = 0
	epoch_count = 0
	theta_prev = np.ones((X.shape[1], 1))
	theta_current = np.zeros((X.shape[1], 1))

	sum_cost_prev = 100
	sum_cost_curr = 0

	avg_cost_prev = 100
	avg_cost_curr = 0

	count_1000 = 0

	max_epoch = 100000

	is_break = False
	while True:
		if is_break:
			break
		# print(f"Epoch = {epoch_count}")
		epoch_count+=1
		if(epoch_count > max_epoch):
			print(f"Max epoch - {epoch_count - 1} reached")
			break
		for b in range(0, m//r + 1):
			

			if b == m//r:
				rem = Y.shape[0] - m//r * r
				if rem == 0:
					continue
				Xb = X[b*r:]
				Yb = Y[b*r:]
			else:
				rem = r
				Xb = X[b*r:b*r+r]
				Yb = Y[b*r:b*r+r]

			new_cost = cost(Xb, Yb, theta_current)
			sum_cost_curr += new_cost

			if itr_count == 1000 * count_1000:
				avg_cost_curr = sum_cost_curr/1000

				print(f"Epoch = {epoch_count}, Last 1000 iteration avg cost diff= {abs(avg_cost_prev - avg_cost_curr)}" )
				count_1000+= 1
				sum_cost_curr = 0

				if abs(avg_cost_curr - avg_cost_prev) < epsilon:
					print(f"Average cost converged")
					is_break = True
					break

				avg_cost_prev = avg_cost_curr
				avg_cost_curr = 0
				

			itr_count+=1
			
			cost_arr.append(new_cost)
			theta0_arr.append(theta_current[0][0])
			theta1_arr.append(theta_current[1][0])
			theta2_arr.append(theta_current[2][0])
				
			

			h = np.dot(Xb, theta_prev)
			theta_current = theta_prev - (1 / rem) * eta *  np.dot(np.transpose(Xb), (h - Yb))
			theta_prev = theta_current.copy()

			# new_cost = cost(X, Y, theta_current)
			# sum_cost_curr += new_cost

	print(itr_count, epoch_count)
	return {"theta_opt": theta_current, "epoch_count": epoch_count, "itr_count": itr_count, "final_cost": new_cost, "theta0_arr": theta0_arr, "theta1_arr": theta1_arr, "theta2_arr": theta2_arr}


def partA(sample_count, theta):
	data = sample(sample_count, theta)
	X_sample = data["X"]
	Y_sample = data["Y"]
	combined_data = data["C"]


	np.savetxt('sample.csv', combined_data, delimiter=',', fmt='%10.5f')
	print("Sampling done. \n Saved to sample.csv")

	# delete unused data
	del data
	del combined_data

	return {"X":X_sample, "Y":Y_sample}


def partB(X, Y, batch_size):
	start = time.time()
	sgd_data =  SGD(X, Y, batch_size)
	end = time.time()
	print(f"SGD take {(end - start)} time")

	theta_opt = sgd_data["theta_opt"]
	epoch_count = sgd_data["epoch_count"]
	itr_count = sgd_data["itr_count"]
	final_cost = sgd_data["final_cost"]

	print(f"Optimal theta = \n{theta_opt}")
	print(f"Epoch = {epoch_count}")
	print(f"Iteration = {itr_count}")
	print(f"Final cost = {final_cost}")


	return sgd_data

def partC(theta_opt, test_data):
	x_test = test_data[:,(0,1,2)]
	y_test = test_data[:,3].reshape((-1,1))

	cost_test = cost(x_test, y_test, theta_opt)
	
	return cost_test
	

def partD(theta0_arr, theta1_arr, theta2_arr, theta_opt):

	theta0_center = theta_opt[0][0]
	theta1_center = theta_opt[1][0]
	theta2_center = theta_opt[2][0]

	print(theta0_center, theta1_center, theta2_center)

	# theta0_axis = np.linspace(0, theta0_center + 1, 100)
	# theta1_axis = np.linspace(0, theta1_center + 1, 100)
	# theta2_axis = np.linspace(0, theta1_center + 1, 100)


	fig = plt.figure()
	ax = plt.axes(projection='3d')

	ax.set_xlim3d(0, theta0_center + 1)
	ax.set_ylim3d(0, theta1_center + 1)
	ax.set_zlim3d(0, theta2_center + 1)

	
	ax.set_xlabel("θ0", fontweight='bold')
	ax.set_ylabel("θ1", fontweight='bold')
	ax.set_zlabel("θ2", fontweight='bold')


	# plt.plot(theta0_arr, theta1_arr, theta2_arr, color="red")
	def animate(i, theta0_arr, theta1_arr, theta2_arr):
		line, = plt.plot(theta0_arr[:i], theta1_arr[:i], theta2_arr[:i], color="red")
		return line,

	anim = animation.FuncAnimation(fig, animate, frames=len(theta0_arr), fargs=(theta0_arr, theta1_arr, theta2_arr), interval=1)

	plt.show()





def main(test_file, eta, sample_count, theta, batch_size):
	test_data = loadData(test_file)["test_data"]

	#---------part(a)------------------
	print("-"*10)
	print("part(a)")
	print("-"*10)
	data = partA(sample_count, theta)
	X = data["X"]
	Y = data["Y"]


	#---------part(b)------------------
	print("part(b)")
	print("-"*10)
	sgd_data = partB(X, Y, batch_size)
	theta_opt = sgd_data["theta_opt"]

	theta0_arr = sgd_data["theta0_arr"]
	theta1_arr = sgd_data["theta1_arr"]
	theta2_arr = sgd_data["theta2_arr"]





	#---------part(c)------------------
	print("part(c)")
	print("-"*10)
	
	cost_test = partC(theta_opt, test_data)
	print(f"Cost on test data using learned theta = {cost_test}")

	cost_original = partC(theta, test_data)
	print(f"Cost on test data using original theta = {cost_original}")


	#---------part(d)------------------
	partD(theta0_arr, theta1_arr, theta2_arr, theta_opt)


	



if __name__ == "__main__":
	try:
		# test file, batch_size, eta(optional), sample points(optional),
		test_file = sys.argv[1]
		batch_size = int(sys.argv[2])

		eta = 0.001
		sample_count = 1000000
		theta = np.array([[3],[1],[2]])
		if len(sys.argv) > 3:
			eta = float(sys.argv[3])

		if len(sys.argv) > 4:
			sample_count = int(sys.argv[4])

		print(f"Learning rate = {eta}")
		print(f"Sample count = {sample_count}")
		print(f"Theta = \n {theta}")
		print(f"Batch size = {batch_size}")

		main(test_file, eta, sample_count, theta, batch_size)
	except Exception as e:
		print(e)
		msg = '''
=>python3 q2.py test_file batch_size eta(optional) sample_points(optional)
		'''
		print(msg)

